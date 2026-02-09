# -*- coding: utf-8 -*-
"""
记忆条目存储引擎 (SQLite 版)
- 字段全部可选（topic/keywords/summary/body），至少一个非空
- SQLite 实时持久化 + keyword_index 倒排索引表
- 精确 + 模糊搜索
- 时间范围过滤
- 可选自动关键词提取（jieba）
- 多智能体共享（source 标识）
- 新增字段: entry_type, category, date_str, access_count, last_accessed,
            importance, supersedes
"""

import json
import time
import re
import os
import sqlite3
from collections import defaultdict

# ========================
# jieba 按需加载（只在 auto_extract_keywords=True 时使用）
# ========================
_jieba = None


def _ensure_jieba():
    global _jieba
    if _jieba is None:
        import jieba
        _jieba = jieba
    return _jieba


# 停用词集合（与 main_v2.py 保持一致）
STOPWORDS = {
    "。（", "）。", "，"", ""，", "》，", "，《", "）（", "——",
    "表示", "进行", "没有", "可以", "已经", "其中", "不是",
    "就是", "这个", "那个", "什么", "他们", "我们", "自己",
    "应该", "目前", "如果", "通过", "之后", "以及", "以来",
    "因为", "所以", "但是", "而且", "或者", "对于",
    "关于", "根据", "按照", "由于", "虽然", "不过", "然而",
    "还是", "仍然", "只是", "也是", "并且", "同时", "这样",
    "那样", "如何", "怎么", "为什么", "怎样", "哪些", "那些",
    "这些", "一些", "很多", "非常", "比较", "相关", "其他",
    "需要", "成为", "认为", "包括", "来看", "看来", "这是",
    "记者", "报道", "据悉", "了解", "介绍", "方面", "情况",
    "问题", "工作", "发展", "建设", "活动", "地区", "国家",
    "上午", "下午", "昨天", "今天", "明天", "去年", "今年",
    "明年", "上半年", "下半年",
}

RE_NOISE = re.compile(
    r'^(\d+\.?\d*%?|'
    r'\d{4}年?\d{0,2}月?\d{0,2}日?|'
    r'[a-zA-Z]|'
    r'[\u3000\xa0\s]+)$'
)


def extract_keywords_jieba(text, min_len=2):
    """
    从文本中自动提取关键词（jieba cut_for_search + 停用词过滤）。
    用于 auto_extract_keywords=True 时的兜底提取。
    """
    if not text or not isinstance(text, str):
        return []

    jieba = _ensure_jieba()
    words = jieba.cut_for_search(text)

    clean = set()
    for w in words:
        if len(w) < min_len:
            continue
        if w in STOPWORDS:
            continue
        if RE_NOISE.match(w):
            continue
        clean.add(w)

    # 子串去重：短词是长词的子串时只保留长词
    sorted_words = sorted(clean, key=len, reverse=True)
    result = []
    for word in sorted_words:
        if not any(word in kept and word != kept for kept in result):
            result.append(word)

    return result


# ========================
# SQLite 建表 SQL
# ========================

_CREATE_ENTRIES_TABLE = """
CREATE TABLE IF NOT EXISTS entries (
    id            TEXT PRIMARY KEY,
    entry_type    TEXT NOT NULL DEFAULT 'knowledge',
    topic         TEXT,
    summary       TEXT,
    body          TEXT,
    source        TEXT,
    category      TEXT,
    timestamp     REAL NOT NULL,
    date_str      TEXT,
    keywords_json TEXT,
    access_count  INTEGER DEFAULT 0,
    last_accessed REAL,
    importance    INTEGER DEFAULT 3,
    supersedes    TEXT
)
"""

_CREATE_KEYWORD_INDEX_TABLE = """
CREATE TABLE IF NOT EXISTS keyword_index (
    keyword   TEXT NOT NULL,
    entry_id  TEXT NOT NULL,
    PRIMARY KEY (keyword, entry_id)
)
"""

_CREATE_INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_type       ON entries(entry_type)",
    "CREATE INDEX IF NOT EXISTS idx_date       ON entries(date_str)",
    "CREATE INDEX IF NOT EXISTS idx_category   ON entries(entry_type, category)",
    "CREATE INDEX IF NOT EXISTS idx_source     ON entries(source)",
    "CREATE INDEX IF NOT EXISTS idx_timestamp  ON entries(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_importance ON entries(importance)",
    "CREATE INDEX IF NOT EXISTS idx_keyword    ON keyword_index(keyword)",
]

_CREATE_META_TABLE = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
)
"""


class MemoryStore:
    """
    多智能体共享记忆存储引擎 (SQLite 版)。

    支持两种模式:
    1. SQLite 持久化模式: MemoryStore(db_path="./data/memory.db")
       - 所有写入实时持久化到 SQLite
       - 启动时自动加载

    2. 纯内存模式（向后兼容）: MemoryStore()
       - 使用 :memory: SQLite（内存数据库）
       - 仍可通过 save()/load() 做 JSON 备份/恢复

    写入:
        store.add(topic="中美贸易", keywords=["关税","出口"], summary="...", body="...")
        store.add(summary="一段笔记", auto_extract_keywords=True)
        store.add(keywords=["概念A","概念B"])

    检索:
        store.search_exact(["关税"])           -> {entry_id: 命中数}
        store.search_fuzzy(["贸易"])           -> {entry_id: 匹配分}
        store.filter_by_time(ids, time_recent=24)  -> 最近24小时的条目
    """

    def __init__(self, db_path=None):
        """
        初始化存储引擎。

        参数:
            db_path: str | None
                - None: 使用内存 SQLite（向后兼容，行为与旧 dict 版一致）
                - 路径字符串: 使用文件 SQLite，实时持久化
        """
        if db_path is None:
            self._db_path = ":memory:"
        else:
            self._db_path = db_path
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        # 开启 WAL 模式提高并发性能（仅文件模式有效）
        if self._db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._init_tables()
        self._next_id = self._load_next_id()

    def _init_tables(self):
        """创建表和索引"""
        cur = self._conn.cursor()
        cur.execute(_CREATE_ENTRIES_TABLE)
        cur.execute(_CREATE_KEYWORD_INDEX_TABLE)
        cur.execute(_CREATE_META_TABLE)
        for idx_sql in _CREATE_INDICES:
            cur.execute(idx_sql)
        self._conn.commit()

    def _load_next_id(self):
        """从 meta 表或已有数据恢复 next_id"""
        cur = self._conn.cursor()
        # 先查 meta 表
        row = cur.execute("SELECT value FROM meta WHERE key='next_id'").fetchone()
        if row:
            return int(row[0])
        # 从现有条目推断
        row = cur.execute(
            "SELECT id FROM entries ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if row:
            last_id = row[0]
            try:
                num = int(last_id.split("_")[1])
                return num
            except (IndexError, ValueError):
                pass
        return 0

    def _save_next_id(self):
        """保存 next_id 到 meta 表"""
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('next_id', ?)",
            (str(self._next_id),)
        )

    def _generate_id(self):
        self._next_id += 1
        return f"mem_{self._next_id:06d}"

    # ========================
    # 写入
    # ========================

    def add(self, topic=None, keywords=None, summary=None, body=None,
            source=None, auto_extract_keywords=False, timestamp=None,
            entry_type="knowledge", category=None, date_str=None,
            importance=3, supersedes=None):
        """
        写入一条记忆条目。

        参数:
            topic:    str | None   -- 主题
            keywords: list[str] | None -- 关键词列表（AI 直接给的）
            summary:  str | None   -- 摘要描述
            body:     str | None   -- 正文（重点+细节）
            source:   str | None   -- 来源标识（哪个智能体写入的）
            auto_extract_keywords: bool -- 无 keywords 时是否用 jieba 自动提取
            timestamp: float | None -- 写入时间戳，默认 time.time()
            entry_type: str -- 条目类型: 'knowledge'/'experience'/'log'
            category: str | None -- 分类（自由文本）
            date_str: str | None -- 日期字符串（如 '2025-01-15'）
            importance: int -- 重要度 1~5，默认3
            supersedes: str | None -- 指向被取代的旧条目 ID

        返回: str -- entry_id
        """
        # 校验：至少一个内容字段非空
        has_content = any([topic, keywords, summary, body])
        if not has_content:
            raise ValueError("至少需要一个非空字段（topic/keywords/summary/body）")

        # 校验 importance 范围
        importance = max(1, min(5, int(importance)))

        entry_id = self._generate_id()
        ts = timestamp if timestamp is not None else time.time()

        # 处理关键词
        final_keywords = list(keywords) if keywords else []

        # 自动提取关键词（兜底）
        if auto_extract_keywords and not final_keywords:
            text_parts = []
            if topic:
                text_parts.append(topic)
            if summary:
                text_parts.append(summary)
            if body:
                text_parts.append(body[:500])
            if text_parts:
                final_keywords = extract_keywords_jieba(" ".join(text_parts))

        keywords_json = json.dumps(final_keywords, ensure_ascii=False) if final_keywords else None

        # 处理 supersedes: 降低被取代条目的 importance
        if supersedes:
            self._handle_supersedes(supersedes)

        # 插入 entries 表
        self._conn.execute("""
            INSERT INTO entries (id, entry_type, topic, summary, body, source,
                                 category, timestamp, date_str, keywords_json,
                                 access_count, last_accessed, importance, supersedes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?)
        """, (entry_id, entry_type, topic, summary, body, source,
              category, ts, date_str, keywords_json,
              importance, supersedes))

        # 建倒排索引
        self._index_entry_sql(entry_id, final_keywords, topic)

        # 保存 next_id
        self._save_next_id()
        self._conn.commit()

        return entry_id

    def _handle_supersedes(self, old_entry_id):
        """处理记忆再巩固: 被取代的旧条目 importance 减 1"""
        row = self._conn.execute(
            "SELECT importance FROM entries WHERE id=?", (old_entry_id,)
        ).fetchone()
        if row:
            old_importance = max(1, row[0] - 1)
            self._conn.execute(
                "UPDATE entries SET importance=? WHERE id=?",
                (old_importance, old_entry_id)
            )

    def _index_entry_sql(self, entry_id, keywords, topic):
        """为一条记忆条目建立 keyword_index 倒排索引"""
        pairs = []
        for kw in (keywords or []):
            pairs.append((kw, entry_id))
        if topic:
            pairs.append((topic, entry_id))
        if pairs:
            self._conn.executemany(
                "INSERT OR IGNORE INTO keyword_index (keyword, entry_id) VALUES (?, ?)",
                pairs
            )

    # ========================
    # 读取 / 删除
    # ========================

    def get(self, entry_id):
        """按 ID 获取记忆条目，不存在返回 None。返回 dict 格式与旧版兼容。"""
        row = self._conn.execute(
            "SELECT * FROM entries WHERE id=?", (entry_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def remove(self, entry_id):
        """删除一条记忆条目，返回是否成功"""
        row = self._conn.execute(
            "SELECT id FROM entries WHERE id=?", (entry_id,)
        ).fetchone()
        if row is None:
            return False

        # 删除倒排索引
        self._conn.execute(
            "DELETE FROM keyword_index WHERE entry_id=?", (entry_id,)
        )
        # 删除条目
        self._conn.execute("DELETE FROM entries WHERE id=?", (entry_id,))
        self._conn.commit()
        return True

    # ========================
    # 搜索
    # ========================

    def search_exact(self, keywords):
        """
        精确匹配：通过 keyword_index 表查找包含指定关键词的条目。

        参数:  keywords: list[str]
        返回:  dict[str, int] -- {entry_id: 命中关键词数量}，按命中数降序
        """
        if not keywords:
            return {}

        hits = defaultdict(int)
        for kw in keywords:
            rows = self._conn.execute(
                "SELECT entry_id FROM keyword_index WHERE keyword=?", (kw,)
            ).fetchall()
            for row in rows:
                hits[row[0]] += 1

        return dict(sorted(hits.items(), key=lambda x: x[1], reverse=True))

    def search_fuzzy(self, keywords):
        """
        模糊匹配：前缀 + 子串包含。

        "贸易" -> 命中 "贸易战", "国际贸易", "贸易协定"
        "特朗" -> 命中 "特朗普"

        参数:  keywords: list[str]
        返回:  dict[str, float] -- {entry_id: 模糊匹配得分}，按分数降序
        """
        if not keywords:
            return {}

        # 获取所有索引关键词（用于模糊匹配）
        all_index_keys = [row[0] for row in
                          self._conn.execute(
                              "SELECT DISTINCT keyword FROM keyword_index"
                          ).fetchall()]

        hits = defaultdict(float)

        for query_kw in keywords:
            for idx_kw in all_index_keys:
                if query_kw == idx_kw:
                    continue    # 精确匹配由 search_exact 处理

                score = 0.0

                if query_kw in idx_kw:
                    # 查询词是索引词的子串："贸易" in "贸易战"
                    score = len(query_kw) / len(idx_kw)
                elif idx_kw in query_kw:
                    # 索引词是查询词的子串
                    score = len(idx_kw) / len(query_kw) * 0.8

                if score > 0:
                    rows = self._conn.execute(
                        "SELECT entry_id FROM keyword_index WHERE keyword=?",
                        (idx_kw,)
                    ).fetchall()
                    for row in rows:
                        hits[row[0]] = max(hits[row[0]], score)

        return dict(sorted(hits.items(), key=lambda x: x[1], reverse=True))

    def filter_by_time(self, entry_ids, time_range=None, time_recent=None):
        """
        时间范围过滤（硬过滤，范围外直接排除）。

        参数:
            entry_ids:   可迭代的 entry_id 集合
            time_range:  tuple(start_ts, end_ts) -- 精确时间范围
            time_recent: float -- 最近 N 小时（与 time_range 二选一）
        返回:
            list[str] -- 过滤后的 entry_id 列表
        """
        if time_range is None and time_recent is None:
            return list(entry_ids)

        now = time.time()
        if time_range:
            start_ts, end_ts = time_range
        else:
            start_ts = now - time_recent * 3600
            end_ts = now

        entry_id_list = list(entry_ids)
        if not entry_id_list:
            return []

        # 分批查询（SQLite 变量数限制）
        result = []
        batch_size = 500
        for i in range(0, len(entry_id_list), batch_size):
            batch = entry_id_list[i:i + batch_size]
            placeholders = ",".join(["?"] * len(batch))
            rows = self._conn.execute(
                f"SELECT id FROM entries WHERE id IN ({placeholders}) "
                f"AND timestamp >= ? AND timestamp <= ?",
                batch + [start_ts, end_ts]
            ).fetchall()
            result.extend(row[0] for row in rows)

        return result

    def search_by_source(self, source):
        """按来源智能体过滤，返回 entry_id 列表"""
        rows = self._conn.execute(
            "SELECT id FROM entries WHERE source=?", (source,)
        ).fetchall()
        return [row[0] for row in rows]

    # ========================
    # 时间权重（反比函数，importance 感知版）
    # ========================

    @staticmethod
    def compute_time_weight(entry_timestamp, decay_lambda=0.1, now=None,
                            importance=3):
        """
        反比时间衰减权重（importance 感知版）。

        公式:  weight = 1 / (1 + adjusted_lambda * age_days)
               adjusted_lambda = decay_lambda * 3.0 / importance

        importance=3 时等价于旧公式 1/(1+lambda*age_days)（完全向后兼容）
        importance=5 时衰减更慢（重要的事记得更久）
        importance=1 时衰减更快（不重要的事更快遗忘）

        lambda=0.1, importance=3 时:
          0天: 1.00    1天: 0.91    7天: 0.59
          30天: 0.25   100天: 0.09  365天: 0.027

        lambda=0.1, importance=5 时:
          0天: 1.00    1天: 0.94    7天: 0.70
          30天: 0.36   100天: 0.14  365天: 0.044

        参数:
            entry_timestamp: float -- 条目写入时间戳
            decay_lambda: float -- 衰减系数，越大衰减越快
            now: float | None -- 当前时间戳，默认 time.time()
            importance: int -- 重要度 1~5，默认3
        返回:
            float -- 0~1 之间的时间权重
        """
        if now is None:
            now = time.time()
        age_days = max(0, (now - entry_timestamp) / 86400.0)
        # importance=3 时 adjusted_lambda = decay_lambda（与旧公式完全一致）
        # importance=5 时 adjusted_lambda = decay_lambda * 0.6（衰减更慢）
        # importance=1 时 adjusted_lambda = decay_lambda * 3.0（衰减更快）
        adjusted_lambda = decay_lambda * 3.0 / max(1, importance)
        return 1.0 / (1.0 + adjusted_lambda * age_days)

    # ========================
    # 提取强化: access_count / last_accessed
    # ========================

    def increment_access(self, entry_ids):
        """
        批量递增条目的 access_count 并更新 last_accessed。
        在 query() 返回结果后由上层调用。

        参数: entry_ids: list[str] -- 被查询命中的条目 ID 列表
        """
        if not entry_ids:
            return
        now = time.time()
        for eid in entry_ids:
            self._conn.execute(
                "UPDATE entries SET access_count = access_count + 1, "
                "last_accessed = ? WHERE id = ?",
                (now, eid)
            )
        self._conn.commit()

    # ========================
    # 重要度管理
    # ========================

    def set_importance(self, entry_id, level):
        """
        设置条目的重要度级别。

        参数:
            entry_id: str -- 条目 ID
            level: int -- 重要度 1~5
        返回:
            bool -- 是否成功（条目是否存在）
        """
        level = max(1, min(5, int(level)))
        result = self._conn.execute(
            "UPDATE entries SET importance=? WHERE id=?",
            (level, entry_id)
        )
        self._conn.commit()
        return result.rowcount > 0

    # ========================
    # 列表 / 统计 / 工具
    # ========================

    def list_entries(self, source_filter=None, entry_type=None):
        """
        列出所有条目（可按 source 或 entry_type 过滤）。

        返回: list[dict]
        """
        conditions = []
        params = []
        if source_filter:
            conditions.append("source = ?")
            params.append(source_filter)
        if entry_type:
            conditions.append("entry_type = ?")
            params.append(entry_type)

        sql = "SELECT * FROM entries"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY timestamp DESC"

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_all_keywords(self):
        """返回倒排索引中所有关键词列表"""
        rows = self._conn.execute(
            "SELECT DISTINCT keyword FROM keyword_index"
        ).fetchall()
        return [row[0] for row in rows]

    def get_stats(self):
        """返回存储统计信息"""
        total = self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        total_keys = self._conn.execute(
            "SELECT COUNT(DISTINCT keyword) FROM keyword_index"
        ).fetchone()[0]
        sources_rows = self._conn.execute(
            "SELECT DISTINCT source FROM entries WHERE source IS NOT NULL"
        ).fetchall()
        sources = sorted(row[0] for row in sources_rows)

        return {
            "total_entries": total,
            "total_index_keys": total_keys,
            "sources": sources,
        }

    def get_entries_since(self, since_timestamp):
        """
        获取指定时间戳之后的所有条目。
        用于 consolidate(rebuild_from_recent=N) 场景。

        参数: since_timestamp: float -- 起始时间戳
        返回: list[dict]
        """
        rows = self._conn.execute(
            "SELECT * FROM entries WHERE timestamp >= ? ORDER BY timestamp",
            (since_timestamp,)
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def __len__(self):
        return self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    # ========================
    # 持久化 (兼容旧接口 + SQLite 原生)
    # ========================

    def save(self, path):
        """
        保存到 JSON 文件（向后兼容）。
        注意: SQLite 模式下数据已实时持久化，此方法主要用于导出备份。
        """
        entries = {}
        rows = self._conn.execute("SELECT * FROM entries").fetchall()
        for row in rows:
            d = self._row_to_dict(row)
            entries[d["id"]] = d

        data = {
            "entries": entries,
            "next_id": self._next_id,
        }
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        """
        从 JSON 文件加载（向后兼容）。
        将 JSON 中的条目导入到当前 SQLite 数据库中。
        返回: bool -- 是否成功
        """
        if not os.path.exists(path):
            return False

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data.get("entries", {})
        next_id = data.get("next_id", 0)

        # 清空现有数据
        self._conn.execute("DELETE FROM keyword_index")
        self._conn.execute("DELETE FROM entries")

        # 导入
        for entry_id, entry in entries.items():
            keywords = entry.get("keywords") or []
            keywords_json = json.dumps(keywords, ensure_ascii=False) if keywords else None

            self._conn.execute("""
                INSERT OR REPLACE INTO entries
                (id, entry_type, topic, summary, body, source, category,
                 timestamp, date_str, keywords_json,
                 access_count, last_accessed, importance, supersedes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry_id,
                entry.get("entry_type", "knowledge"),
                entry.get("topic"),
                entry.get("summary"),
                entry.get("body"),
                entry.get("source"),
                entry.get("category"),
                entry.get("timestamp", time.time()),
                entry.get("date_str"),
                keywords_json,
                entry.get("access_count", 0),
                entry.get("last_accessed"),
                entry.get("importance", 3),
                entry.get("supersedes"),
            ))

            self._index_entry_sql(entry_id, keywords, entry.get("topic"))

        self._next_id = next_id
        self._save_next_id()
        self._conn.commit()

        return True

    # ========================
    # 内部工具
    # ========================

    def _row_to_dict(self, row):
        """将 sqlite3.Row 转为与旧版兼容的 dict 格式"""
        keywords_json = row["keywords_json"]
        keywords = json.loads(keywords_json) if keywords_json else None

        d = {
            "id": row["id"],
            "topic": row["topic"],
            "keywords": keywords,
            "summary": row["summary"],
            "body": row["body"],
            "source": row["source"],
            "timestamp": row["timestamp"],
            # 新字段
            "entry_type": row["entry_type"],
            "category": row["category"],
            "date_str": row["date_str"],
            "access_count": row["access_count"],
            "last_accessed": row["last_accessed"],
            "importance": row["importance"],
            "supersedes": row["supersedes"],
        }
        return d

    @property
    def entries(self):
        """
        向后兼容属性: 返回所有条目的 dict 视图。
        警告: 这会加载所有条目到内存，仅用于兼容旧代码。
        大数据量时请用 list_entries() 或 SQL 查询代替。
        """
        rows = self._conn.execute("SELECT * FROM entries").fetchall()
        return {row["id"]: self._row_to_dict(row) for row in rows}

    @property
    def inverted_index(self):
        """
        向后兼容属性: 返回倒排索引的 dict 视图。
        警告: 仅用于兼容旧代码。
        """
        rows = self._conn.execute(
            "SELECT keyword, entry_id FROM keyword_index"
        ).fetchall()
        idx = defaultdict(set)
        for row in rows:
            idx[row[0]].add(row[1])
        return dict(idx)

    def close(self):
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        """析构时自动关闭连接"""
        try:
            self.close()
        except Exception:
            pass
