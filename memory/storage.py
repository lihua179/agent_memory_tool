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
import datetime
import calendar
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


# 名词性词性白名单（用于 jieba.analyse.extract_tags 的 allowPOS 参数）
# n=普通名词, ns=地名, nt=机构名, nz=其他专名, nrt=翻译人名
# nr=人名, eng=英文词(如Transformer), vn=动名词(如"运算","编程")
# l=习用语(如"自然语言"), i=成语/习语, j=简称
_NOUN_POS_ALLOW = ('n', 'ns', 'nt', 'nz', 'nr', 'nrt',
                   'eng', 'vn', 'l', 'i', 'j')


def extract_nouns_jieba(text, top_k=20, min_len=2):
    """
    从文本中按 TF-IDF 权重提取名词性关键词（高质量概念词）。

    与 extract_keywords_jieba 的区别：
    - 使用 jieba.analyse.extract_tags 的 TF-IDF 排序 + allowPOS 词性过滤
    - 只保留名词性词语（n/ns/nt/nz/nr/eng/vn/l/i/j），过滤动词/副词/介词等噪声
    - 返回带权重的元组列表，按 TF-IDF 权重降序排列
    - 适用于自动模式下向共现矩阵注入高质量概念词

    参数:
        text: str - 输入文本
        top_k: int - 最多提取多少个关键词（默认20）
        min_len: int - 最小词长（过滤单字）

    返回:
        list[tuple(str, float)] - [(词, 权重), ...] 按 TF-IDF 权重降序
    """
    if not text or not isinstance(text, str):
        return []

    analyse = _ensure_jieba_analyse()

    # TF-IDF 提取 + 词性白名单过滤（多取一些，后续还要过滤停用词和噪声）
    raw_tags = analyse.extract_tags(
        text, topK=top_k * 2, withWeight=True, allowPOS=_NOUN_POS_ALLOW)

    # 二次过滤：停用词 + 噪声正则 + 最小词长
    filtered = []
    for word, weight in raw_tags:
        if len(word) < min_len:
            continue
        if word in STOPWORDS:
            continue
        if RE_NOISE.match(word):
            continue
        filtered.append((word, weight))

    # 子串去重：短词是长词子串时只保留长词（保留权重较高者）
    sorted_by_len = sorted(filtered, key=lambda x: len(x[0]), reverse=True)
    result = []
    for word, weight in sorted_by_len:
        if not any(word in kept_w and word != kept_w for kept_w, _ in result):
            result.append((word, weight))

    # 按权重降序排列，取 top_k
    result.sort(key=lambda x: x[1], reverse=True)
    return result[:top_k]


# ========================
# jieba TF-IDF 按需加载
# ========================
_jieba_analyse = None


def _ensure_jieba_analyse():
    global _jieba_analyse
    if _jieba_analyse is None:
        import jieba.analyse
        _jieba_analyse = jieba.analyse
    return _jieba_analyse


def extract_keywords_weighted(text, top_k=25, long_threshold=80, min_len=2):
    """
    智能关键词提取（长文本 TF-IDF 带权重排序，短文本退化为平权）。

    短文本 (≤ long_threshold 字): 退化为 extract_keywords_jieba()，权重统一 1.0
    长文本 (> long_threshold 字): 使用 jieba.analyse.extract_tags() TF-IDF 排序

    参数:
        text: str - 输入文本
        top_k: int - 最多提取多少个关键词（默认25）
        long_threshold: int - 长文本阈值（字符数），默认80
        min_len: int - 最小词长

    返回:
        list[tuple(str, float)] — [(词, 权重), ...] 按权重降序
        短文本时权重统一为 1.0
    """
    if not text or not isinstance(text, str):
        return []

    # 短文本：退化为 extract_keywords_jieba，权重均为 1.0
    if len(text) <= long_threshold:
        words = extract_keywords_jieba(text, min_len=min_len)
        return [(w, 1.0) for w in words]

    # 长文本：使用 jieba TF-IDF 提取带权重的关键词
    analyse = _ensure_jieba_analyse()
    raw_tags = analyse.extract_tags(text, topK=top_k * 2, withWeight=True)

    # 过滤：停用词 + 噪声 + 最小词长
    filtered = []
    for word, weight in raw_tags:
        if len(word) < min_len:
            continue
        if word in STOPWORDS:
            continue
        if RE_NOISE.match(word):
            continue
        filtered.append((word, weight))

    # 子串去重：短词是长词的子串时只保留长词（保留权重较高者）
    sorted_by_len = sorted(filtered, key=lambda x: len(x[0]), reverse=True)
    result = []
    for word, weight in sorted_by_len:
        if not any(word in kept_w and word != kept_w for kept_w, _ in result):
            result.append((word, weight))

    # 按权重降序排列，取 top_k
    result.sort(key=lambda x: x[1], reverse=True)
    return result[:top_k]


# ========================
# 时间表达式自动解析器
# ========================

def _parse_cn_num(s):
    """
    中文/阿拉伯数字 → int（支持 0~99）。

    支持格式:
        阿拉伯: "3", "12", "300"（仅取用于时间单位的小数字）
        中文: "三", "十二", "二十三", "两"
    返回 None 表示解析失败。
    """
    if not s:
        return None

    # 阿拉伯数字直接转换
    if s.isdigit():
        return int(s)

    cn_map = {
        "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        "〇": 0, "壹": 1, "贰": 2, "叁": 3, "肆": 4,
        "伍": 5, "陆": 6, "柒": 7, "捌": 8, "玖": 9, "拾": 10,
    }

    # 单字直接查表
    if len(s) == 1 and s in cn_map:
        return cn_map[s]

    # "十X" → 10+X  (e.g. 十二→12)
    if s.startswith("十") or s.startswith("拾"):
        if len(s) == 1:
            return 10
        rest = s[1:]
        if rest in cn_map:
            return 10 + cn_map[rest]
        return None

    # "X十" → X*10  (e.g. 二十→20)
    # "X十Y" → X*10+Y  (e.g. 二十三→23)
    for i, ch in enumerate(s):
        if ch in ("十", "拾") and i > 0:
            tens_ch = s[:i]
            units_part = s[i + 1:]
            if tens_ch in cn_map:
                tens = cn_map[tens_ch] * 10
                if not units_part:
                    return tens
                if units_part in cn_map:
                    return tens + cn_map[units_part]
            return None

    return None


def _resolve_year_prefix(prefix, now_dt):
    """
    解析年份前缀 → year int。

    支持: 今年/去年/前年/大前年/2024年/2024
    参数:
        prefix: str - "今年"/"去年"/"前年"/"大前年"/"2024年"/"2024"/""
        now_dt: datetime.date - 当前日期
    返回: int - 年份，解析失败返回 now_dt.year
    """
    if not prefix:
        return now_dt.year

    prefix = prefix.rstrip("年")

    if prefix == "今":
        return now_dt.year
    if prefix == "去":
        return now_dt.year - 1
    if prefix == "前":
        return now_dt.year - 2
    if prefix == "大前":
        return now_dt.year - 3

    # 4位阿拉伯年份: "2024"
    if prefix.isdigit() and len(prefix) == 4:
        return int(prefix)

    return now_dt.year


def _month_ts_range(year, month):
    """
    返回某年某月的时间戳范围: (月初 00:00:00 ts, 月末 23:59:59 ts)。
    """
    first_day = datetime.datetime(year, month, 1, 0, 0, 0)
    _, last_day_num = calendar.monthrange(year, month)
    last_day = datetime.datetime(year, month, last_day_num, 23, 59, 59)
    return first_day.timestamp(), last_day.timestamp()


def _day_ts_range(dt):
    """
    返回某天的时间戳范围: (当天 00:00:00 ts, 当天 23:59:59 ts)。
    参数: dt: datetime.date 或 datetime.datetime
    """
    if isinstance(dt, datetime.date) and not isinstance(dt, datetime.datetime):
        dt = datetime.datetime(dt.year, dt.month, dt.day)
    start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end = dt.replace(hour=23, minute=59, second=59, microsecond=0)
    return start.timestamp(), end.timestamp()


def _calc_relative_range(today, n, unit):
    """
    计算"前N天/周/月/年"的时间范围。

    返回: (start_ts, end_ts) — 从 N 个单位前到今天 23:59:59

    参数:
        today: datetime.date - 当前日期
        n: int - 数量
        unit: str - "天"/"日"/"周"/"星期"/"月"/"年"
    """
    today_dt = datetime.datetime(today.year, today.month, today.day)
    end_ts = today_dt.replace(hour=23, minute=59, second=59).timestamp()

    if unit in ("天", "日"):
        start_dt = today_dt - datetime.timedelta(days=n)
    elif unit in ("周", "星期"):
        start_dt = today_dt - datetime.timedelta(weeks=n)
    elif unit == "月":
        year = today.year
        month = today.month - n
        while month <= 0:
            month += 12
            year -= 1
        # 处理日期溢出（如 3月31日 → 前1个月 → 2月28日）
        _, max_day = calendar.monthrange(year, month)
        day = min(today.day, max_day)
        start_dt = datetime.datetime(year, month, day)
    elif unit == "年":
        year = today.year - n
        month = today.month
        day = today.day
        # 处理闰年边界（2月29日回退到非闰年）
        try:
            start_dt = datetime.datetime(year, month, day)
        except ValueError:
            start_dt = datetime.datetime(year, month, 28)
    else:
        return None

    start_ts = start_dt.replace(hour=0, minute=0, second=0).timestamp()
    return start_ts, end_ts


def _calc_simple_time(today, expr, now_dt=None):
    """
    解析简单时间表达式。

    支持: 今天/昨天/前天/大前天/上周/上上周/上个月/上上个月/本周/本月/今年/去年/前年

    参数:
        today: datetime.date - 当前日期
        expr: str - 时间表达式
        now_dt: datetime.datetime | None - 当前时间
    返回:
        (start_ts, end_ts) | None
    """
    today_dt = datetime.datetime(today.year, today.month, today.day)

    if expr == "今天":
        return _day_ts_range(today_dt)

    if expr == "昨天":
        return _day_ts_range(today_dt - datetime.timedelta(days=1))

    if expr == "前天":
        return _day_ts_range(today_dt - datetime.timedelta(days=2))

    if expr == "大前天":
        return _day_ts_range(today_dt - datetime.timedelta(days=3))

    if expr == "上周":
        # 上周一到上周日
        # weekday(): 0=周一, 6=周日
        days_since_monday = today.weekday()
        this_monday = today_dt - datetime.timedelta(days=days_since_monday)
        last_monday = this_monday - datetime.timedelta(weeks=1)
        last_sunday = last_monday + datetime.timedelta(days=6)
        return (last_monday.replace(hour=0, minute=0, second=0).timestamp(),
                last_sunday.replace(hour=23, minute=59, second=59).timestamp())

    if expr == "上上周":
        days_since_monday = today.weekday()
        this_monday = today_dt - datetime.timedelta(days=days_since_monday)
        target_monday = this_monday - datetime.timedelta(weeks=2)
        target_sunday = target_monday + datetime.timedelta(days=6)
        return (target_monday.replace(hour=0, minute=0, second=0).timestamp(),
                target_sunday.replace(hour=23, minute=59, second=59).timestamp())

    if expr == "本周":
        days_since_monday = today.weekday()
        this_monday = today_dt - datetime.timedelta(days=days_since_monday)
        this_sunday = this_monday + datetime.timedelta(days=6)
        return (this_monday.replace(hour=0, minute=0, second=0).timestamp(),
                this_sunday.replace(hour=23, minute=59, second=59).timestamp())

    if expr in ("上个月", "上月"):
        year = today.year
        month = today.month - 1
        if month <= 0:
            month += 12
            year -= 1
        return _month_ts_range(year, month)

    if expr == "上上个月":
        year = today.year
        month = today.month - 2
        while month <= 0:
            month += 12
            year -= 1
        return _month_ts_range(year, month)

    if expr == "本月":
        return _month_ts_range(today.year, today.month)

    if expr == "今年":
        start = datetime.datetime(today.year, 1, 1, 0, 0, 0)
        end = datetime.datetime(today.year, 12, 31, 23, 59, 59)
        return start.timestamp(), end.timestamp()

    if expr == "去年":
        y = today.year - 1
        start = datetime.datetime(y, 1, 1, 0, 0, 0)
        end = datetime.datetime(y, 12, 31, 23, 59, 59)
        return start.timestamp(), end.timestamp()

    if expr == "前年":
        y = today.year - 2
        start = datetime.datetime(y, 1, 1, 0, 0, 0)
        end = datetime.datetime(y, 12, 31, 23, 59, 59)
        return start.timestamp(), end.timestamp()

    return None


# ---- 时间表达式置信度评估 ----
# 用于区分"搜索约束"与"内容描述中的时间词"

# 相对时间词集合——天然更偏向查询意图
_RELATIVE_TIME_EXPRS = {
    "今天", "昨天", "前天", "大前天",
    "上周", "上上周", "本周",
    "上个月", "上上个月", "上月", "本月",
    "今年", "去年", "前年",
    "最近",
}

# 查询意图动词
_RE_QUERY_VERB = re.compile(
    r'(找|搜|查|看看|帮我|回忆|想想|回顾|搜索|查找|查看|查询|检索|翻翻)\s*$'
)

# 描述性动词（时间词前出现这些 → 大概率是内容描述）
_RE_DESC_VERB = re.compile(
    r'(提到|说了|写了|记录了|描述|讲述|讨论|谈到|介绍|分析了|引用了|列举了|统计了|预测)\s*$'
)

# "里/中/内" 容器模式（"文章里""报告中" → 表示引述，时间词更可能是内容）
# 检查时间词前方近距离（10字以内）是否有此模式
_RE_CONTAINER_NEARBY = re.compile(r'[里中内]')


def _is_in_quotes(text, start, end):
    """
    检查 text[start:end] 是否被引号或书名号包裹。

    支持：
        中文引号:  "" '' 《》 〈〉 「」
        英文引号:  "" ''
    """
    quote_pairs = [
        ('\u201c', '\u201d'),   # 中文双引号 "" 
        ('\u2018', '\u2019'),   # 中文单引号 ''
        ('\u300a', '\u300b'),   # 书名号 《》
        ('\u3008', '\u3009'),   # 尖括号 〈〉
        ('\u300c', '\u300d'),   # 直角引号 「」
        ('"', '"'),             # 英文双引号
        ("'", "'"),             # 英文单引号
    ]
    for open_q, close_q in quote_pairs:
        # 在 start 之前找最近的 open_q
        open_pos = text.rfind(open_q, 0, start)
        if open_pos == -1:
            continue
        # 在 end 之后找最近的 close_q
        close_pos = text.find(close_q, end)
        if close_pos != -1:
            return True
    return False


def _compute_time_confidence(text, match, time_expr):
    """
    计算时间表达式作为"搜索约束"（而非"内容描述"）的置信度。

    参数:
        text: str       — 完整的用户输入文本
        match: re.Match — 正则匹配结果
        time_expr: str  — 匹配到的时间表达式文本

    返回:
        (score: float, signals: list[str])
        score 范围 [0.0, 1.0]，越高越可能是搜索约束
    """
    score = 0.5  # 基础分
    signals = []

    pos = match.start()
    before = text[:pos]          # 时间词之前的文本
    after = text[match.end():]   # 时间词之后的文本

    # 去除前导空白后判断
    before_stripped = before.rstrip()

    # ============ 正向信号 ============

    # 1. 句首（前面无实质内容，或只有标点/空白）
    if len(before_stripped) == 0 or re.match(r'^[，。！？、；：\s]*$', before_stripped):
        score += 0.3
        signals.append("+句首")

    # 2. 前有查询动词
    if _RE_QUERY_VERB.search(before_stripped):
        score += 0.3
        signals.append("+查询动词")

    # 3. 后跟"的"字
    if after.startswith("的"):
        score += 0.1
        signals.append("+的字后缀")

    # 4. 短文本（<40字，更像查询指令）
    if len(text) < 40:
        score += 0.15
        signals.append("+短文本")

    # 5. 相对时间表达（今天/昨天/上周等天然偏向查询）
    #    对于"最近N天"等组合，检查是否以"最近"开头
    is_relative = (time_expr in _RELATIVE_TIME_EXPRS or
                   time_expr.startswith("最近"))
    if is_relative:
        score += 0.1
        signals.append("+相对时间")

    # ============ 负向信号 ============

    # 6. 在引号/书名号内
    if _is_in_quotes(text, match.start(), match.end()):
        score -= 0.5
        signals.append("-引号内")

    # 7. 前有描述性动词
    if _RE_DESC_VERB.search(before_stripped):
        score -= 0.35
        signals.append("-描述动词")

    # 8. 前方近距离（10字内）有"里/中/内"（"文章里""报告中"）
    #    这类容器词暗示前文在引述某个文档/报告的内容
    near_before = before_stripped[-10:] if len(before_stripped) > 10 else before_stripped
    if _RE_CONTAINER_NEARBY.search(near_before):
        score -= 0.25
        signals.append("-里/中/内")

    # 9. 长文本（>120字，更像内容描述而非搜索指令）
    if len(text) > 120:
        score -= 0.1
        signals.append("-长文本")

    return max(0.0, min(1.0, score)), signals


# ---- 正则模式（按优先级从高到低排列） ----

# 月份范围: "今年5月到8月" / "今年5月份到8月份" / "2024年3月~6月"
_RE_MONTH_RANGE = re.compile(
    r'(今|去|前|大前|\d{4})年?'
    r'(\d{1,2}|[一二三四五六七八九十]+)月份?'
    r'[到至~\-]'
    r'(\d{1,2}|[一二三四五六七八九十]+)月份?'
)

# 年+月: "今年5月" / "去年12月" / "2024年3月"
_RE_YEAR_MONTH = re.compile(
    r'(今|去|前|大前|\d{4})年'
    r'(\d{1,2}|[一二三四五六七八九十]+)月份?'
)

# 年+月+日: "今年5月3日" / "2024年3月15日"
_RE_YEAR_MONTH_DAY = re.compile(
    r'(今|去|前|大前|\d{4})年'
    r'(\d{1,2}|[一二三四五六七八九十]+)月'
    r'(\d{1,2}|[一二三四五六七八九十]+)[日号]'
)

# "前N天/周/月/年": "前两个月" / "前3天" / "前一年"
_RE_BEFORE_N = re.compile(
    r'前(\d+|[一二两三四五六七八九十百]+)个?(天|日|周|星期|月|年)'
)

# "N天/周/月/年前": "两个月前" / "3天前" / "一年前"
_RE_N_AGO = re.compile(
    r'(\d+|[一二两三四五六七八九十百]+)个?(天|日|周|星期|月|年)前'
)

# "最近N天/周/月/年": "最近三天" / "最近2个月"
_RE_RECENT_N = re.compile(
    r'最近(\d+|[一二两三四五六七八九十百]+)个?(天|日|周|星期|月|年)'
)

# 简单时间: 今天/昨天/前天/大前天/上周/上上周/本周/上个月/上上个月/上月/本月/今年/去年/前年
_RE_SIMPLE = re.compile(
    r'(大前天|前天|昨天|今天|上上周|上周|本周|上上个月|上个月|上月|本月|今年|去年|前年)'
)

# 四位年份: "2024年"（单独出现，表示整年）
_RE_SPEC_YEAR = re.compile(
    r'(\d{4})年'
)


def parse_time_expression(text, now=None, confidence_threshold=0.45):
    """
    从中文自然语言中自动提取时间表达式，拆分为时间约束 + 内容关键词。

    内置置信度评估：通过时间词的位置、上下文动词、引号包裹等启发式规则，
    判断时间词是"搜索约束"还是"内容描述"。置信度低于阈值时，time_range
    返回 None，原文不做删改。

    按优先级依次尝试匹配以下模式（最具体的优先）：
        1. year_month_day: "今年5月3日" / "2024年3月15日"
        2. month_range: "今年5月到8月" / "2024年3月~6月"
        3. year_month: "今年5月" / "去年12月"
        4. before_n: "前两个月" / "前3天"
        5. n_ago: "两个月前" / "3天前"
        6. recent_n: "最近三天" / "最近2个月"
        7. simple: "今天" / "昨天" / "上周" / "去年" 等
        8. spec_year: "2024年"

    参数:
        text: str - 用户原始输入文本
        now: datetime.datetime | None - 当前时间，默认 datetime.datetime.now()
        confidence_threshold: float - 置信度阈值，低于此值则不作为时间约束
            默认 0.45。设为 0.0 可禁用置信度过滤（兼容旧行为）。

    返回:
        dict - {
            "time_range": (start_ts, end_ts) | None,  # 解析出的时间范围（低置信度时为 None）
            "keywords": list[str],      # 去除时间表达式后提取的内容关键词
            "time_expr": str | None,    # 匹配到的时间表达式原文（不受置信度影响）
            "cleaned_text": str,        # 去除时间表达式后的文本（低置信度时保留原文）
            "confidence": float | None, # 时间词作为搜索约束的置信度（无匹配时为 None）
            "confidence_signals": list[str] | None,  # 置信度计算的各信号（调试用）
        }
    """
    if not text or not isinstance(text, str):
        return {
            "time_range": None,
            "keywords": extract_keywords_jieba(text) if text else [],
            "time_expr": None,
            "cleaned_text": text or "",
            "confidence": None,
            "confidence_signals": None,
        }

    if now is None:
        now = datetime.datetime.now()
    today = now.date()

    time_range = None
    time_expr = None
    matched_span = None  # (start, end) in text
    matched_obj = None   # re.Match 对象，用于 confidence 计算

    # ---- 按优先级尝试匹配 ----

    # 1. year_month_day: "今年5月3日"
    if time_range is None:
        m = _RE_YEAR_MONTH_DAY.search(text)
        if m:
            year = _resolve_year_prefix(m.group(1), today)
            month = _parse_cn_num(m.group(2))
            day = _parse_cn_num(m.group(3))
            if month and 1 <= month <= 12 and day and 1 <= day <= 31:
                try:
                    target_dt = datetime.datetime(year, month, day)
                    time_range = _day_ts_range(target_dt)
                    time_expr = m.group(0)
                    matched_span = m.span()
                    matched_obj = m
                except ValueError:
                    pass

    # 2. month_range: "今年5月到8月"
    if time_range is None:
        m = _RE_MONTH_RANGE.search(text)
        if m:
            year = _resolve_year_prefix(m.group(1), today)
            month_start = _parse_cn_num(m.group(2))
            month_end = _parse_cn_num(m.group(3))
            if (month_start and month_end and
                    1 <= month_start <= 12 and 1 <= month_end <= 12):
                start_ts, _ = _month_ts_range(year, month_start)
                _, end_ts = _month_ts_range(year, month_end)
                time_range = (start_ts, end_ts)
                time_expr = m.group(0)
                matched_span = m.span()
                matched_obj = m

    # 3. year_month: "今年5月"
    if time_range is None:
        m = _RE_YEAR_MONTH.search(text)
        if m:
            year = _resolve_year_prefix(m.group(1), today)
            month = _parse_cn_num(m.group(2))
            if month and 1 <= month <= 12:
                time_range = _month_ts_range(year, month)
                time_expr = m.group(0)
                matched_span = m.span()
                matched_obj = m

    # 4. before_n: "前两个月"
    if time_range is None:
        m = _RE_BEFORE_N.search(text)
        if m:
            n = _parse_cn_num(m.group(1))
            unit = m.group(2)
            if n and n > 0:
                result = _calc_relative_range(today, n, unit)
                if result:
                    time_range = result
                    time_expr = m.group(0)
                    matched_span = m.span()
                    matched_obj = m

    # 5. n_ago: "两个月前"
    if time_range is None:
        m = _RE_N_AGO.search(text)
        if m:
            n = _parse_cn_num(m.group(1))
            unit = m.group(2)
            if n and n > 0:
                result = _calc_relative_range(today, n, unit)
                if result:
                    time_range = result
                    time_expr = m.group(0)
                    matched_span = m.span()
                    matched_obj = m

    # 6. recent_n: "最近三天"
    if time_range is None:
        m = _RE_RECENT_N.search(text)
        if m:
            n = _parse_cn_num(m.group(1))
            unit = m.group(2)
            if n and n > 0:
                result = _calc_relative_range(today, n, unit)
                if result:
                    time_range = result
                    time_expr = m.group(0)
                    matched_span = m.span()
                    matched_obj = m

    # 7. simple: "今天"/"昨天"/"上周" 等
    if time_range is None:
        m = _RE_SIMPLE.search(text)
        if m:
            result = _calc_simple_time(today, m.group(1), now)
            if result:
                time_range = result
                time_expr = m.group(0)
                matched_span = m.span()
                matched_obj = m

    # 8. spec_year: "2024年"
    if time_range is None:
        m = _RE_SPEC_YEAR.search(text)
        if m:
            year = int(m.group(1))
            if 1900 <= year <= 2100:
                start = datetime.datetime(year, 1, 1, 0, 0, 0)
                end = datetime.datetime(year, 12, 31, 23, 59, 59)
                time_range = (start.timestamp(), end.timestamp())
                time_expr = m.group(0)
                matched_span = m.span()
                matched_obj = m

    # ---- 置信度评估 ----
    confidence = None
    confidence_signals = None

    if matched_obj is not None and time_range is not None:
        confidence, confidence_signals = _compute_time_confidence(
            text, matched_obj, time_expr
        )
        # 低于阈值 → 不作为搜索约束
        if confidence < confidence_threshold:
            time_range = None
            matched_span = None   # 不做 cleaned_text 删除

    # ---- 构建 cleaned_text（去除匹配到的时间表达式） ----
    if matched_span:
        cleaned = text[:matched_span[0]] + text[matched_span[1]:]
        # 清理可能残留的连接词/标点
        cleaned = re.sub(r'^[的，,\s]+', '', cleaned)
        cleaned = re.sub(r'[的，,\s]+$', '', cleaned)
        # 清理中间可能出现的多余空白
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
        cleaned = cleaned.strip()
    else:
        cleaned = text.strip()

    # ---- 提取关键词（从 cleaned_text 中提取） ----
    keywords = extract_keywords_jieba(cleaned) if cleaned else []

    return {
        "time_range": time_range,
        "keywords": keywords,
        "time_expr": time_expr,
        "cleaned_text": cleaned,
        "confidence": confidence,
        "confidence_signals": confidence_signals,
    }


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
