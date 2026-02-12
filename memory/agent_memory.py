# -*- coding: utf-8 -*-
"""
AgentMemory - 多智能体共享记忆系统统一 API

这是整个记忆系统的唯一对外接口。封装了：
- 共现矩阵构建与管理 (IncrementalCooccurrence)
- PPMI / 条件概率计算 (probability)
- 时间衰减 (DecayManager)
- 记忆条目存储 (MemoryStore, SQLite 实时持久化)
- 多级检索 (MemoryRetriever)
- 隐藏链推理 (chain)

三种记忆类型:
    knowledge — 事实、概念、稳定知识（语义记忆）
    experience — 经验教训、模式总结（经验记忆）
    log — 活动事件、发生了什么（事件记忆）

用法:
    from memory import AgentMemory

    # 初始化（SQLite 持久化模式）
    am = AgentMemory(data_dir="./memory_data")

    # 1. 摄入文档（构建知识图谱）
    am.ingest_document("特朗普宣布对中国加征关税...")
    am.ingest_documents_from_csv("news.csv")

    # 2. 写入记忆条目（AI 智能体写入结构化记忆）
    entry_id = am.write(topic="中美贸易摩擦", keywords=["关税","出口"],
                        summary="...", importance=4)

    # 3. 查询记忆（注入 prompt 上下文）
    result = am.query(keywords=["关税","出口"], depth="standard", token_budget=1000)
    print(result["prompt_text"])  # 直接注入到 agent prompt

    # 4. 链推理（复杂问题分析）
    chain = am.find_chain(["特朗普", "A股"])

    # 5. 记忆再巩固（纠正旧知识）
    new_id = am.write(topic="新发现", keywords=["AI"],
                      importance=5, supersedes=old_id)

    # 6. 睡眠整理
    am.consolidate(min_cooccurrence=2, rebuild_from_recent=90)

    # 7. 持久化（矩阵+配置，条目已实时存入SQLite）
    am.save("./memory_data")
    am.load("./memory_data")
"""

import os
import json
import time
import re
import numpy as np
from scipy.sparse import save_npz, load_npz

from .cooccurrence import IncrementalCooccurrence
from .probability import compute_ppmi_matrix, compute_conditional_prob_matrix
from .decay import DecayManager
from .storage import MemoryStore, extract_keywords_jieba, extract_nouns_jieba, parse_time_expression
from .retriever import MemoryRetriever
from .chain import discover_hidden_chain, discover_hidden_chain_with_evidence


class AgentMemory:
    """
    多智能体共享长期关联记忆系统 - 统一接口。

    三层服务模式:
        Layer 1: 每轮对话自动注入关联记忆（query）
        Layer 2: 复杂问题按需链推理（find_chain）
        Layer 3: 日志记忆（log / recall_by_date / recall_by_range / summarize）

    三种记忆类型:
        knowledge — 事实、概念、稳定知识（语义记忆）
        experience — 经验教训、模式总结（经验记忆）
        log — 活动事件、发生了什么（事件记忆）

    三大脑启发机制:
        提取强化 — query 命中的条目自动递增 access_count（统计，不影响排名）
        重要度分级 — importance 1~5 影响时间衰减速率和排名权重
        记忆再巩固 — supersedes 指向旧条目，旧条目自动降级

    参数:
        data_dir: str | None - 数据目录路径。若指定，SQLite 实时持久化到
            data_dir/memory.db；若 None，使用纯内存模式（向后兼容）
        max_dim: int - 共现矩阵最大维度（词汇表上限）
        min_cooccurrence: int - 剪枝阈值（共现次数低于此值的关系被过滤）
        decay_rate: float - 共现矩阵衰减率（每次保留 1-decay_rate）
        decay_interval: int - 每处理多少文档执行一次衰减
        time_decay_lambda: float - 记忆条目时间衰减系数 lambda
            (反比函数 1/(1+lambda*days))
    """

    def __init__(self, data_dir=None, max_dim=100000, min_cooccurrence=5,
                 decay_rate=0.005, decay_interval=500,
                 time_decay_lambda=0.1):
        # 配置
        self._data_dir = data_dir
        self._config = {
            "max_dim": max_dim,
            "min_cooccurrence": min_cooccurrence,
            "decay_rate": decay_rate,
            "decay_interval": decay_interval,
            "time_decay_lambda": time_decay_lambda,
        }

        # 核心组件
        self._cooccurrence = IncrementalCooccurrence(max_dim=max_dim)
        self._decay = DecayManager(decay_rate=decay_rate,
                                   decay_interval=decay_interval)

        # SQLite 存储: data_dir 模式下实时持久化，否则纯内存
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "memory.db")
            self._store = MemoryStore(db_path=db_path)
        else:
            self._store = MemoryStore()

        # 矩阵缓存（需要 rebuild_matrices 后才可用）
        self._pruned_matrix = None
        self._ppmi_matrix = None

        # 检索器（延迟初始化，rebuild_matrices 后自动创建）
        self._retriever = None

        # 状态跟踪
        self._matrices_dirty = True  # 标记矩阵是否需要重建
        self._last_rebuild_docs = 0

        # Episodic Layer（日志记忆）索引
        self._date_index = {}        # {"2026-02-09": ["mem_001", ...]} 按时间正序
        self._category_index = {}    # {"工作": {"mem_001", ...}}
        self._episodic_ids = set()   # 标记哪些 entry 是日志条目

    # ================================================================
    # 文档摄入（构建共现知识）
    # ================================================================

    def ingest_document(self, text, min_word_len=2):
        """
        摄入一篇文档文本，学习词共现关系。

        参数:
            text: str - 文档原始文本
            min_word_len: int - 最小词长（过滤单字）
        """
        if not text or not isinstance(text, str):
            return

        words = extract_keywords_jieba(text, min_len=min_word_len)
        if len(words) < 2:
            return

        self._cooccurrence.add_document(words)
        self._decay.maybe_decay(self._cooccurrence)
        self._matrices_dirty = True

    def ingest_documents_from_csv(self, csv_path, content_column="content",
                                  min_word_len=2, progress_callback=None):
        """
        从 CSV 文件批量摄入文档。

        参数:
            csv_path: str - CSV 文件路径
            content_column: str - 内容列名
            min_word_len: int - 最小词长
            progress_callback: callable | None - 进度回调 fn(current, total)
                (注意: 不要在管道/子进程中使用 tqdm，会导致缓冲区溢出)

        返回:
            dict - {"docs_processed": int, "docs_skipped": int, "time_seconds": float}
        """
        import pandas as pd

        t0 = time.time()
        df = pd.read_csv(csv_path)

        if content_column not in df.columns:
            raise ValueError(f"CSV 中未找到列 '{content_column}', "
                             f"可用列: {list(df.columns)}")

        docs_processed = 0
        docs_skipped = 0
        total = len(df)

        for idx, row in df.iterrows():
            text = row.get(content_column)
            if not isinstance(text, str) or len(text.strip()) < 10:
                docs_skipped += 1
                continue

            self.ingest_document(text, min_word_len=min_word_len)
            docs_processed += 1

            if progress_callback and docs_processed % 100 == 0:
                progress_callback(docs_processed, total)

        elapsed = time.time() - t0
        return {
            "docs_processed": docs_processed,
            "docs_skipped": docs_skipped,
            "time_seconds": round(elapsed, 2),
        }

    # ================================================================
    # 记忆条目 CRUD
    # ================================================================

    def write(self, topic=None, keywords=None, summary=None, body=None,
              source=None, auto_extract_keywords=False, timestamp=None,
              entry_type="knowledge", category=None, importance=3,
              supersedes=None):
        """
        写入一条记忆条目。

        参数:
            topic: str | None - 主题
            keywords: list[str] | None - 关键词列表（AI 直接给的）
            summary: str | None - 摘要
            body: str | None - 正文
            source: str | None - 来源标识（哪个 agent 写入的）
            auto_extract_keywords: bool - 无 keywords 时是否用 jieba 自动提取
            timestamp: float | None - 写入时间戳，默认当前时间
            entry_type: str - 条目类型 'knowledge'/'experience'/'log'
            category: str | None - 分类（自由文本）
            importance: int - 重要度 1~5，默认3
            supersedes: str | None - 指向被取代的旧条目 ID（记忆再巩固）

        返回:
            str - entry_id
        """
        ts = timestamp if timestamp is not None else time.time()
        date_str = self._ts_to_date_str(ts)

        entry_id = self._store.add(
            topic=topic, keywords=keywords,
            summary=summary, body=body,
            source=source,
            auto_extract_keywords=auto_extract_keywords,
            timestamp=ts,
            entry_type=entry_type,
            category=category,
            date_str=date_str,
            importance=importance,
            supersedes=supersedes,
        )

        # 同步学习: 将条目的关键词注入共现矩阵（Channel B）
        entry = self._store.get(entry_id)
        if entry:
            ai_gave_keywords = bool(keywords)  # AI 主动给了 keywords
            entry_keywords = entry.get("keywords") or []

            if ai_gave_keywords and len(entry_keywords) >= 2:
                # AI 主动给关键词 → 信任 AI，只用 keywords 做共现
                self._cooccurrence.add_keywords(entry_keywords)
                self._matrices_dirty = True
            elif not ai_gave_keywords:
                # 自动模式 → 从 topic + keywords + summary 提取名词性概念词
                # extract_nouns_jieba 返回 [(word, weight), ...] 按 TF-IDF 权重降序
                text_parts = []
                if entry.get("topic"):
                    text_parts.append(entry["topic"])
                if entry_keywords:
                    text_parts.extend(entry_keywords)
                if entry.get("summary"):
                    text_parts.append(entry["summary"])
                combined_text = " ".join(text_parts)
                noun_pairs = extract_nouns_jieba(combined_text, top_k=20)
                nouns = [w for w, _ in noun_pairs]
                # 将自动提取的 keywords 也并入（它们通过 auto_extract 得到，
                # 可能包含非名词，但已经过 extract_keywords_jieba 过滤）
                noun_set = set(nouns)
                for kw in entry_keywords:
                    if kw not in noun_set:
                        noun_set.add(kw)
                        nouns.append(kw)
                if len(nouns) >= 2:
                    self._cooccurrence.add_keywords(nouns)
                    self._matrices_dirty = True

        return entry_id

    def read(self, entry_id):
        """
        读取一条记忆条目。

        参数:
            entry_id: str - 条目 ID
        返回:
            dict | None - 条目内容，不存在返回 None
        """
        return self._store.get(entry_id)

    def remove(self, entry_id):
        """
        删除一条记忆条目。

        参数:
            entry_id: str - 条目 ID
        返回:
            bool - 是否删除成功
        """
        return self._store.remove(entry_id)

    def list_entries(self, source_filter=None, entry_type=None):
        """
        列出所有记忆条目（可按来源或类型过滤）。

        参数:
            source_filter: str | None - 只列某个 agent 的条目
            entry_type: str | None - 只列某种类型 'knowledge'/'experience'/'log'
        返回:
            list[dict] - 条目列表
        """
        return self._store.list_entries(
            source_filter=source_filter,
            entry_type=entry_type,
        )

    # ================================================================
    # 查询（主 API — 用于 agent prompt 注入）
    # ================================================================

    def query(self, keywords=None, user_input=None, depth="standard",
              token_budget=1000, time_recent=None, time_range=None,
              source_filter=None, top_n_expand=10, top_n_entries=10,
              chain_fuzzy=False, auto_parse_time=False,
              long_threshold=80, core_ratio=0.2, important_ratio=0.4):
        """
        多级记忆查询，返回可直接注入 prompt 的结果。

        参数:
            keywords: list[str] | None - AI 给的关键词（优先）
            user_input: str | None - 用户原始输入（兜底 jieba 提取）
            depth: str - "fast" / "standard" / "deep"
                fast: 精确+模糊匹配（毫秒级）
                standard: +PPMI 关联扩展（百毫秒级）
                deep: +隐藏链推理（秒级）
            token_budget: int - prompt 注入的最大 token 数
            time_recent: float | None - 只搜最近 N 小时内的记忆
            time_range: tuple(start_ts, end_ts) | None - 精确时间范围
            source_filter: str | None - 只搜某个 agent 的记忆
            top_n_expand: int - 关联扩展取 top-N 关联词
            top_n_entries: int - 最终返回最多几条
            chain_fuzzy: bool - 隐藏链推理词是否也做模糊匹配（默认 False）
            auto_parse_time: bool - 是否自动从 user_input 中解析时间表达式
                为 True 时，会自动识别中文时间表达式（如"上周""前两个月"），
                将其转换为 time_range 约束，并从搜索关键词中剔除时间词。
                仅在 user_input 非空且未显式传入 time_range/time_recent 时生效。
            long_threshold: int - 长文本模式触发阈值（字符数），默认 80
                超过此长度的 user_input 将使用 TF-IDF 分级提取关键词，
                避免关键词爆炸导致搜索噪声。
            core_ratio: float - 长文本模式下核心词占提取总词数的比例，默认 0.2
            important_ratio: float - 长文本模式下重要词占提取总词数的比例，默认 0.4

        返回:
            dict - {
                "prompt_text": str,        # 拼好的注入文本（<=token_budget）
                "matched_entries": list,   # 命中条目详情
                "expanded_keywords": list, # 关联扩展词
                "chain": dict | None,      # 推理链（deep 才有）
                "search_stats": dict,      # 搜索统计
                "parsed_time": dict | None, # 时间解析结果（auto_parse_time=True 时）
            }
        """
        # ---- 时间表达式自动解析 ----
        parsed_time_info = None
        if auto_parse_time and user_input and time_range is None and time_recent is None:
            parsed = parse_time_expression(user_input)
            if parsed["time_range"] is not None:
                time_range = parsed["time_range"]
                user_input = parsed["cleaned_text"]
                parsed_time_info = parsed

        # 确保检索器可用
        self._ensure_retriever()

        result = self._retriever.retrieve(
            user_input=user_input,
            keywords=keywords,
            depth=depth,
            token_budget=token_budget,
            time_range=time_range,
            time_recent=time_recent,
            time_decay=self._config["time_decay_lambda"],
            source_filter=source_filter,
            top_n_expand=top_n_expand,
            top_n_entries=top_n_entries,
            chain_fuzzy=chain_fuzzy,
            long_threshold=long_threshold,
            core_ratio=core_ratio,
            important_ratio=important_ratio,
        )

        # 附加时间解析信息
        result["parsed_time"] = parsed_time_info

        # 提取强化: 递增命中条目的 access_count
        matched = result.get("matched_entries", [])
        if matched:
            hit_ids = [e["id"] for e in matched if "id" in e]
            if hit_ids:
                self._store.increment_access(hit_ids)

        return result

    # ================================================================
    # 链推理（按需 — 用于复杂问题分析）
    # ================================================================

    def find_chain(self, anchor_words, top_k_candidates=50, alpha=0.85,
                   min_edge_weight=0.1, top_n=10, with_evidence=False):
        """
        隐藏链推理：给定锚点词，发现中间关联概念和链路。

        参数:
            anchor_words: list[str] - 锚点词（至少 2 个）
            top_k_candidates: int - PPR 候选词数量
            alpha: float - PPR 阻尼因子
            min_edge_weight: float - 图最小边权阈值
            top_n: int - 返回隐藏词数量
            with_evidence: bool - 是否附加记忆条目文本证据

        返回:
            dict - {
                "hidden_words": [...],
                "chains": [...],
                "anchors_found": [...],
                "anchors_missing": [...],
                "evidence": [...] (仅 with_evidence=True),
            }
        """
        self._ensure_matrices()

        if self._ppmi_matrix is None:
            return {
                "hidden_words": [], "chains": [],
                "anchors_found": [], "anchors_missing": list(anchor_words),
                "error": "PPMI 矩阵未构建，请先摄入文档并调用 rebuild_matrices()",
            }

        if with_evidence:
            return discover_hidden_chain_with_evidence(
                self._ppmi_matrix,
                self._cooccurrence.vocab_dict,
                anchor_words,
                memory_store=self._store,
                top_k_candidates=top_k_candidates,
                alpha=alpha,
                min_edge_weight=min_edge_weight,
                top_n=top_n,
                fallback_matrix=self._pruned_matrix,
            )
        else:
            return discover_hidden_chain(
                self._ppmi_matrix,
                self._cooccurrence.vocab_dict,
                anchor_words,
                top_k_candidates=top_k_candidates,
                alpha=alpha,
                min_edge_weight=min_edge_weight,
                top_n=top_n,
                fallback_matrix=self._pruned_matrix,
            )

    # ================================================================
    # 日志记忆 — Episodic Layer
    # ================================================================

    @staticmethod
    def _ts_to_date_str(ts):
        """时间戳 → 本地日期字符串 'YYYY-MM-DD'"""
        return time.strftime("%Y-%m-%d", time.localtime(ts))

    @staticmethod
    def _ts_to_time_str(ts):
        """时间戳 → 本地时间字符串 'HH:MM'"""
        return time.strftime("%H:%M", time.localtime(ts))

    @staticmethod
    def _date_str_to_ts_range(date_str):
        """日期字符串 → (当天 00:00:00 时间戳, 次日 00:00:00 时间戳)"""
        import datetime
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        start = dt.timestamp()
        end = start + 86400
        return start, end

    def _index_episodic(self, entry_id, date_str, category):
        """将日志条目索引到日期索引和分类索引中。"""
        # 日期索引（保持列表，写入时已是时间正序）
        if date_str not in self._date_index:
            self._date_index[date_str] = []
        self._date_index[date_str].append(entry_id)

        # 分类索引
        if category:
            if category not in self._category_index:
                self._category_index[category] = set()
            self._category_index[category].add(entry_id)

        # 标记为日志条目
        self._episodic_ids.add(entry_id)

    def log(self, content, detail=None, category=None, tags=None,
            source=None, timestamp=None, auto_extract_keywords=True,
            importance=3):
        """
        写入一条日志（活动/事件记录）。

        与 write() 的区别：
        - entry_type 自动设为 'log'
        - 自动建立日期索引，支持 recall_by_date() 按天查询
        - 自动建立分类索引，支持按 category 过滤
        - 默认开启关键词自动提取
        - 关键词写入共现矩阵（与知识图谱融合）

        参数:
            content: str - 活动概要（必填）
            detail: str | None - 详细内容（可选，长文本）
            category: str | None - 分类标签（自由文本，如"工作""生活"）
            tags: list[str] | None - 手动标签（与自动提取的关键词合并）
            source: str | None - 来源标识
            timestamp: float | None - 发生时间，默认当前时间
            auto_extract_keywords: bool - 是否自动从 content 提取关键词
            importance: int - 重要度 1~5，默认3

        返回:
            str - entry_id
        """
        if not content or not isinstance(content, str):
            raise ValueError("content 不能为空")

        ts = timestamp if timestamp is not None else time.time()

        # 合并 tags（手动标签优先，自动提取兜底）
        final_keywords = list(tags) if tags else None

        # 字段映射：category→topic, content→summary, detail→body
        entry_id = self.write(
            topic=category,
            keywords=final_keywords,
            summary=content,
            body=detail,
            source=source,
            auto_extract_keywords=auto_extract_keywords,
            timestamp=ts,
            entry_type="log",
            category=category,
            importance=importance,
        )

        # 建立日志专属索引
        date_str = self._ts_to_date_str(ts)
        self._index_episodic(entry_id, date_str, category)

        return entry_id

    def recall_by_date(self, date_str):
        """
        按日期查询日志 — 纯时间流查询。

        参数:
            date_str: str - 日期字符串 "YYYY-MM-DD"

        返回:
            dict - {
                "date": str,
                "entries": [
                    {"id": str, "time": str, "category": str,
                     "content": str, "detail": str|None,
                     "tags": list|None, "source": str|None},
                    ...
                ],  # 按时间正序排列
                "count": int,
                "categories": {str: int},  # 当天分类统计
            }
        """
        entry_ids = self._date_index.get(date_str, [])

        entries = []
        cat_counts = {}
        for eid in entry_ids:
            entry = self._store.get(eid)
            if entry is None:
                continue
            category = entry.get("topic")
            entries.append({
                "id": eid,
                "time": self._ts_to_time_str(entry["timestamp"]),
                "category": category,
                "content": entry.get("summary"),
                "detail": entry.get("body"),
                "tags": entry.get("keywords"),
                "source": entry.get("source"),
            })
            if category:
                cat_counts[category] = cat_counts.get(category, 0) + 1

        return {
            "date": date_str,
            "entries": entries,
            "count": len(entries),
            "categories": cat_counts,
        }

    def recall_by_range(self, start_date, end_date, category=None,
                        keyword=None, source_filter=None):
        """
        按时间段查询日志（可叠加分类/关键词过滤）。

        参数:
            start_date: str - 起始日期 "YYYY-MM-DD"（含）
            end_date: str - 结束日期 "YYYY-MM-DD"（含）
            category: str | None - 只看某分类
            keyword: str | None - 叠加关键词过滤（content/detail/tags 中包含）
            source_filter: str | None - 只看某来源

        返回:
            dict - {
                "range": [start_date, end_date],
                "total_count": int,
                "days": {
                    "2026-02-06": [entry_dicts...],
                    "2026-02-07": [entry_dicts...],
                    ...
                },
                "categories": {str: int},  # 整个时间段的分类统计
            }
        """
        import datetime
        dt_start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        dt_end = datetime.datetime.strptime(end_date, "%Y-%m-%d")

        days = {}
        total_count = 0
        all_cat_counts = {}

        # 遍历日期范围内的每一天
        dt_cur = dt_start
        while dt_cur <= dt_end:
            ds = dt_cur.strftime("%Y-%m-%d")
            entry_ids = self._date_index.get(ds, [])

            day_entries = []
            for eid in entry_ids:
                entry = self._store.get(eid)
                if entry is None:
                    continue

                # 分类过滤
                entry_cat = entry.get("topic")
                if category and entry_cat != category:
                    continue

                # 来源过滤
                if source_filter and entry.get("source") != source_filter:
                    continue

                # 关键词过滤（在 content/detail/tags 中搜索）
                if keyword:
                    found = False
                    if entry.get("summary") and keyword in entry["summary"]:
                        found = True
                    if not found and entry.get("body") and keyword in entry["body"]:
                        found = True
                    if not found and entry.get("keywords"):
                        for kw in entry["keywords"]:
                            if keyword in kw or kw in keyword:
                                found = True
                                break
                    if not found:
                        continue

                day_entries.append({
                    "id": eid,
                    "time": self._ts_to_time_str(entry["timestamp"]),
                    "category": entry_cat,
                    "content": entry.get("summary"),
                    "detail": entry.get("body"),
                    "tags": entry.get("keywords"),
                    "source": entry.get("source"),
                })

                if entry_cat:
                    all_cat_counts[entry_cat] = all_cat_counts.get(entry_cat, 0) + 1

            if day_entries:
                days[ds] = day_entries
                total_count += len(day_entries)

            dt_cur += datetime.timedelta(days=1)

        return {
            "range": [start_date, end_date],
            "total_count": total_count,
            "days": days,
            "categories": all_cat_counts,
        }

    def summarize(self, period="week", end_date=None):
        """
        对指定时间段内的日志进行汇总统计。

        返回结构化数据，不做文本生成（留给 AI agent 总结）。

        参数:
            period: str - "day" / "week" / "month"
            end_date: str | None - 截止日期 "YYYY-MM-DD"，默认今天

        返回:
            dict - {
                "period": str,        # "2026-02-03 ~ 2026-02-09"
                "period_type": str,    # "week"
                "total_activities": int,
                "by_category": {str: int},
                "by_day": {"2026-02-03": int, ...},
                "entries": [entry_dicts...],  # 全部条目（供 AI 生成摘要）
            }
        """
        import datetime

        if end_date:
            dt_end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        else:
            dt_end = datetime.datetime.now()

        if period == "day":
            dt_start = dt_end
        elif period == "week":
            dt_start = dt_end - datetime.timedelta(days=6)
        elif period == "month":
            dt_start = dt_end - datetime.timedelta(days=29)
        else:
            raise ValueError(f"period 必须是 'day'/'week'/'month', 收到: {period}")

        start_str = dt_start.strftime("%Y-%m-%d")
        end_str = dt_end.strftime("%Y-%m-%d")

        # 复用 recall_by_range 获取全部条目
        range_result = self.recall_by_range(start_str, end_str)

        # 按天统计数量
        by_day = {}
        all_entries = []
        for ds, day_entries in sorted(range_result["days"].items()):
            by_day[ds] = len(day_entries)
            all_entries.extend(day_entries)

        return {
            "period": f"{start_str} ~ {end_str}",
            "period_type": period,
            "total_activities": range_result["total_count"],
            "by_category": range_result["categories"],
            "by_day": by_day,
            "entries": all_entries,
        }

    # ================================================================
    # 重要度管理 + 记忆再巩固
    # ================================================================

    def set_importance(self, entry_id, level):
        """
        设置条目的重要度级别。

        参数:
            entry_id: str - 条目 ID
            level: int - 重要度 1~5
        返回:
            bool - 是否成功（条目是否存在）
        """
        return self._store.set_importance(entry_id, level)

    # ================================================================
    # 矩阵管理
    # ================================================================

    def rebuild_matrices(self, min_cooccurrence=None):
        """
        重建 PPMI 矩阵（从当前共现矩阵）。

        摄入大量文档后调用此方法更新关联知识。
        也会自动重建检索器。

        参数:
            min_cooccurrence: int | None - 剪枝阈值，None 则使用初始配置值

        返回:
            dict - {
                "vocab_size": int,
                "pruned_nnz": int,
                "ppmi_nnz": int,
                "time_ms": float,
            }
        """
        t0 = time.time()

        thresh = min_cooccurrence if min_cooccurrence is not None \
            else self._config["min_cooccurrence"]

        # 剪枝
        self._pruned_matrix = self._cooccurrence.prune(min_cooccurrence=thresh)

        # PPMI
        self._ppmi_matrix = compute_ppmi_matrix(
            self._pruned_matrix, self._cooccurrence.total_docs)

        # 重建检索器
        self._retriever = MemoryRetriever(
            store=self._store,
            ppmi_matrix=self._ppmi_matrix,
            vocab_dict=self._cooccurrence.vocab_dict,
            cooccurrence_matrix=self._pruned_matrix,
        )

        self._matrices_dirty = False
        self._last_rebuild_docs = self._cooccurrence.total_docs

        elapsed_ms = round((time.time() - t0) * 1000, 2)

        return {
            "vocab_size": self._cooccurrence.vocab_count,
            "pruned_nnz": self._pruned_matrix.nnz,
            "ppmi_nnz": self._ppmi_matrix.nnz,
            "time_ms": elapsed_ms,
        }

    def _ensure_matrices(self):
        """确保矩阵已构建（如果脏了就自动重建）。"""
        if self._matrices_dirty and self._cooccurrence.total_docs > 0:
            self.rebuild_matrices()

    def _ensure_retriever(self):
        """确保检索器已初始化。"""
        if self._retriever is None:
            if self._cooccurrence.total_docs > 0:
                self._ensure_matrices()
            # 即使没有文档，也创建一个基础检索器（只支持 fast 模式）
            if self._retriever is None:
                self._retriever = MemoryRetriever(store=self._store)

    # ================================================================
    # 睡眠整理 — Consolidation
    # ================================================================

    def consolidate(self, min_cooccurrence=None, max_vocab=None,
                    rebuild_from_recent=None):
        """
        睡眠整理: 词汇剪枝 + 矩阵重建 + 持久化。

        模拟人脑睡眠时的记忆整理过程。设计为定期（如每天凌晨）执行。

        流程:
            1. cleanup_vocab() — 从共现矩阵中移除低频词汇
            2. 可选: 从近 N 天的记忆重建共现矩阵（淘汰远期关联）
            3. rebuild_matrices() — 重建 PPMI 矩阵
            4. save() — 持久化矩阵和配置

        参数:
            min_cooccurrence: int | None - 最低共现次数，低于此值的词对被剪枝
                默认使用 config 中的值
            max_vocab: int | None - 词汇表上限，超过时按频率裁剪最低频词
                默认不限制
            rebuild_from_recent: int | None - 只用最近 N 天的记忆条目重建矩阵
                默认 None（使用当前矩阵，不重建）

        返回:
            dict - {
                "vocab_before": int,
                "vocab_after": int,
                "words_removed": int,
                "rebuild_stats": dict,  # rebuild_matrices() 返回值
            }
        """
        vocab_before = self._cooccurrence.vocab_count

        # Step 1: 如果指定了 rebuild_from_recent，从 SQLite 取近 N 天条目重建
        if rebuild_from_recent is not None and rebuild_from_recent > 0:
            cutoff_ts = time.time() - rebuild_from_recent * 86400
            recent_entries = self._store.get_entries_since(cutoff_ts)

            # 重置共现矩阵（保留 vocab 结构，清空计数）
            max_dim = self._config["max_dim"]
            from scipy.sparse import dok_matrix
            self._cooccurrence.matrix = dok_matrix(
                (max_dim, max_dim), dtype=np.float64
            )
            self._cooccurrence.total_docs = 0

            # 用近期条目的关键词重建
            for entry in recent_entries:
                kws = entry.get("keywords")
                if kws and len(kws) >= 2:
                    self._cooccurrence.add_keywords(kws)

        # Step 2: 词汇剪枝
        words_removed = self.cleanup_vocab(
            min_cooccurrence=min_cooccurrence,
            max_vocab=max_vocab,
        )

        # Step 3: 重建矩阵
        thresh = min_cooccurrence if min_cooccurrence is not None \
            else self._config["min_cooccurrence"]
        rebuild_stats = self.rebuild_matrices(min_cooccurrence=thresh)

        # Step 4: 持久化（如果有 data_dir）
        if self._data_dir:
            self.save(self._data_dir)

        vocab_after = self._cooccurrence.vocab_count

        return {
            "vocab_before": vocab_before,
            "vocab_after": vocab_after,
            "words_removed": words_removed,
            "rebuild_stats": rebuild_stats,
        }

    def cleanup_vocab(self, min_cooccurrence=None, max_vocab=None):
        """
        词汇剪枝: 从共现矩阵中移除低频词汇。

        注意: 只影响共现矩阵（关联发现），不影响 SQLite 中的记忆条目。
        类似人脑"遗忘"罕见概念间的关联，但直接提问仍能回忆。

        参数:
            min_cooccurrence: int | None - 共现次数低于此值的词对将被剪枝
            max_vocab: int | None - 词汇表上限

        返回:
            int - 被移除的词汇数量
        """
        if not hasattr(self._cooccurrence, 'remove_words'):
            return 0

        thresh = min_cooccurrence if min_cooccurrence is not None \
            else self._config["min_cooccurrence"]

        # 找出低频词（在矩阵中所有共现计数之和低于阈值的词）
        csr = self._cooccurrence.get_csr_matrix()
        vc = self._cooccurrence.vocab_count

        if vc == 0:
            return 0

        # 计算每个词的总共现频率
        word_freqs = {}
        for idx in range(vc):
            row_sum = csr[idx, :vc].sum()
            col_sum = csr[:vc, idx].sum()
            total = row_sum + col_sum  # 双向共现之和
            word = self._cooccurrence.idx_to_word.get(idx, None)
            if word:
                word_freqs[word] = total

        # 按频率排序，找出要移除的词
        words_to_remove = []
        for word, freq in word_freqs.items():
            if freq < thresh:
                words_to_remove.append(word)

        # 如果有 max_vocab 限制且当前词汇量超出
        if max_vocab and (vc - len(words_to_remove)) > max_vocab:
            # 按频率排序，移除更多低频词
            remaining = {w: f for w, f in word_freqs.items()
                         if w not in words_to_remove}
            sorted_remaining = sorted(remaining.items(), key=lambda x: x[1])
            excess = (vc - len(words_to_remove)) - max_vocab
            for w, f in sorted_remaining[:excess]:
                words_to_remove.append(w)

        if words_to_remove:
            self._cooccurrence.remove_words(words_to_remove)
            self._matrices_dirty = True

        return len(words_to_remove)

    # ================================================================
    # 统计信息
    # ================================================================

    def get_stats(self):
        """
        返回系统综合统计信息。

        返回:
            dict - 包含共现矩阵、记忆存储、衰减管理器、矩阵状态等信息
        """
        store_stats = self._store.get_stats()
        cooc_stats = self._cooccurrence.get_stats()
        decay_info = self._decay.get_info()

        # 日志统计
        episodic_cats = {}
        for eid in self._episodic_ids:
            entry = self._store.get(eid)
            if entry and entry.get("topic"):
                cat = entry["topic"]
                episodic_cats[cat] = episodic_cats.get(cat, 0) + 1

        return {
            "config": self._config,
            "data_dir": self._data_dir,
            "cooccurrence": cooc_stats,
            "store": store_stats,
            "decay": decay_info,
            "matrices": {
                "dirty": self._matrices_dirty,
                "last_rebuild_docs": self._last_rebuild_docs,
                "pruned_nnz": self._pruned_matrix.nnz if self._pruned_matrix is not None else 0,
                "ppmi_nnz": self._ppmi_matrix.nnz if self._ppmi_matrix is not None else 0,
            },
            "episodic": {
                "total_logs": len(self._episodic_ids),
                "total_days": len(self._date_index),
                "categories": episodic_cats,
            },
        }

    # ================================================================
    # 持久化
    # ================================================================

    def save(self, directory):
        """
        将记忆系统状态保存到指定目录。

        保存内容:
            - config.json: 系统配置 + 词汇表 + 衰减状态
            - memory_store.json: 记忆条目备份 (JSON 格式，用于跨系统迁移)
            - cooccurrence.npz: 共现矩阵 (DOK->CSR->npz)
            - pruned.npz: 剪枝后矩阵
            - ppmi.npz: PPMI 矩阵
            - episodic_meta.json: 日志索引

        注意: 使用 data_dir 模式时，记忆条目已实时持久化到 SQLite。
        此方法主要用于保存矩阵和配置状态。

        参数:
            directory: str - 保存目录路径
        """
        os.makedirs(directory, exist_ok=True)

        # 1. 共现矩阵 → CSR → npz
        cooc_csr = self._cooccurrence.get_csr_matrix()
        # 裁剪到实际 vocab_count 大小，避免保存巨大的空矩阵
        vc = self._cooccurrence.vocab_count
        if vc > 0:
            cooc_trimmed = cooc_csr[:vc, :vc]
            save_npz(os.path.join(directory, "cooccurrence.npz"), cooc_trimmed)
        else:
            # 没有数据，保存空标记
            pass

        # 2. 剪枝矩阵
        if self._pruned_matrix is not None and vc > 0:
            pruned_trimmed = self._pruned_matrix[:vc, :vc]
            save_npz(os.path.join(directory, "pruned.npz"), pruned_trimmed)

        # 3. PPMI 矩阵
        if self._ppmi_matrix is not None and vc > 0:
            ppmi_trimmed = self._ppmi_matrix[:vc, :vc]
            save_npz(os.path.join(directory, "ppmi.npz"), ppmi_trimmed)

        # 4. 记忆条目
        store_path = os.path.join(directory, "memory_store.json")
        self._store.save(store_path)

        # 5. 配置 + 词汇表 + 衰减状态 + 元数据
        meta = {
            "config": self._config,
            "vocab_dict": self._cooccurrence.vocab_dict,
            "idx_to_word": {str(k): v for k, v in
                            self._cooccurrence.idx_to_word.items()},
            "vocab_count": self._cooccurrence.vocab_count,
            "total_docs": self._cooccurrence.total_docs,
            "decay_state": self._decay.get_info(),
            "matrices_dirty": self._matrices_dirty,
            "last_rebuild_docs": self._last_rebuild_docs,
            "save_timestamp": time.time(),
        }
        meta_path = os.path.join(directory, "config.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 6. 日志索引（Episodic Layer）
        episodic_meta = {
            "date_index": self._date_index,          # {date_str: [entry_ids]}
            "category_index": {k: sorted(v) for k, v
                               in self._category_index.items()},  # set → list
            "episodic_ids": sorted(self._episodic_ids),            # set → list
        }
        ep_path = os.path.join(directory, "episodic_meta.json")
        with open(ep_path, "w", encoding="utf-8") as f:
            json.dump(episodic_meta, f, ensure_ascii=False, indent=2)

    def load(self, directory):
        """
        从指定目录加载整个记忆系统。

        参数:
            directory: str - 保存目录路径

        返回:
            bool - 是否加载成功
        """
        meta_path = os.path.join(directory, "config.json")
        if not os.path.exists(meta_path):
            return False

        # 1. 加载配置 + 元数据
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self._config = meta.get("config", self._config)

        # 恢复词汇表
        self._cooccurrence.vocab_dict = meta.get("vocab_dict", {})
        # idx_to_word 的 key 需要转回 int
        raw_idx = meta.get("idx_to_word", {})
        self._cooccurrence.idx_to_word = {int(k): v for k, v in raw_idx.items()}
        self._cooccurrence.vocab_count = meta.get("vocab_count", 0)
        self._cooccurrence.total_docs = meta.get("total_docs", 0)

        # 恢复衰减状态
        decay_state = meta.get("decay_state", {})
        self._decay.decay_rate = decay_state.get("decay_rate",
                                                  self._config["decay_rate"])
        self._decay.decay_interval = decay_state.get("decay_interval",
                                                      self._config["decay_interval"])
        self._decay.last_decay_doc = decay_state.get("last_decay_doc", 0)
        self._decay.total_decay_steps = decay_state.get("total_decay_steps", 0)

        self._matrices_dirty = meta.get("matrices_dirty", True)
        self._last_rebuild_docs = meta.get("last_rebuild_docs", 0)

        vc = self._cooccurrence.vocab_count

        # 2. 加载共现矩阵
        cooc_path = os.path.join(directory, "cooccurrence.npz")
        if os.path.exists(cooc_path) and vc > 0:
            cooc_csr = load_npz(cooc_path)
            # 恢复到 max_dim 大小的 DOK 矩阵
            from scipy.sparse import dok_matrix
            max_dim = self._config["max_dim"]
            full_dok = dok_matrix((max_dim, max_dim), dtype=np.float64)
            # 将加载的数据填入
            cooc_coo = cooc_csr.tocoo()
            for r, c, v in zip(cooc_coo.row, cooc_coo.col, cooc_coo.data):
                full_dok[r, c] = v
            self._cooccurrence.matrix = full_dok

        # 3. 加载剪枝矩阵
        pruned_path = os.path.join(directory, "pruned.npz")
        if os.path.exists(pruned_path):
            self._pruned_matrix = load_npz(pruned_path)
            # 需要扩展回 max_dim 大小以保持与 vocab_dict 索引兼容
            if self._pruned_matrix.shape[0] < self._config["max_dim"]:
                from scipy.sparse import csr_matrix as csr_ctor
                max_dim = self._config["max_dim"]
                self._pruned_matrix.resize((max_dim, max_dim))

        # 4. 加载 PPMI 矩阵
        ppmi_path = os.path.join(directory, "ppmi.npz")
        if os.path.exists(ppmi_path):
            self._ppmi_matrix = load_npz(ppmi_path)
            if self._ppmi_matrix.shape[0] < self._config["max_dim"]:
                max_dim = self._config["max_dim"]
                self._ppmi_matrix.resize((max_dim, max_dim))

        # 5. 加载记忆条目
        store_path = os.path.join(directory, "memory_store.json")
        if os.path.exists(store_path):
            self._store.load(store_path)

        # 6. 重建检索器
        if self._ppmi_matrix is not None:
            self._retriever = MemoryRetriever(
                store=self._store,
                ppmi_matrix=self._ppmi_matrix,
                vocab_dict=self._cooccurrence.vocab_dict,
                cooccurrence_matrix=self._pruned_matrix,
            )
        else:
            self._retriever = MemoryRetriever(store=self._store)

        # 7. 加载日志索引（Episodic Layer）
        ep_path = os.path.join(directory, "episodic_meta.json")
        if os.path.exists(ep_path):
            with open(ep_path, "r", encoding="utf-8") as f:
                ep_meta = json.load(f)
            self._date_index = ep_meta.get("date_index", {})
            self._category_index = {
                k: set(v) for k, v in ep_meta.get("category_index", {}).items()
            }
            self._episodic_ids = set(ep_meta.get("episodic_ids", []))

        return True

    # ================================================================
    # 便捷方法
    # ================================================================

    def __repr__(self):
        stats = self.get_stats()
        mode = "sqlite" if self._data_dir else "memory"
        return (
            f"AgentMemory("
            f"mode={mode}, "
            f"docs={stats['cooccurrence']['total_docs']}, "
            f"vocab={stats['cooccurrence']['vocab_size']}, "
            f"entries={stats['store']['total_entries']}, "
            f"ppmi_nnz={stats['matrices']['ppmi_nnz']})"
        )
