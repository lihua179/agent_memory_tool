# -*- coding: utf-8 -*-
"""
多级记忆检索器

四级检索深度：
    fast     — Tier 0 精确 + Tier 1 模糊          （毫秒级）
    standard — Tier 0+1 + Tier 2 PPMI 关联扩展     （百毫秒级）
    deep     — Tier 0+1+2 + Tier 3 隐藏链推理      （秒级）

长文本智能处理：
    当输入超过 long_threshold 字时，使用 TF-IDF 提取带权重的关键词，
    按权重分为核心词/重要词/补充词，各 Tier 使用不同的关键词集合：
        Tier 0 精确: 核心词 + 重要词
        Tier 1 模糊: 仅核心词
        Tier 2 扩展: 仅核心词
        Tier 3 链:   仅核心词
    补充词在评分阶段用于二次验证（加分不扣分）。

分层加载策略（在 token 预算内尽量多加信息）：
    最相关条目: topic → keywords → summary → body（截断）
    次相关条目: topic → keywords → summary
    再次:       topic → keywords
    最后:       关联词提示

时间权重：反比衰减 1/(1+λ*days)
"""

import time
import numpy as np
import tiktoken

from .storage import MemoryStore, extract_keywords_jieba, extract_keywords_weighted
from .chain import discover_hidden_chain


# ========================
# tiktoken 编码器（延迟加载）
# ========================
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")   # GPT-4 / GPT-3.5 通用
    return _encoder


def count_tokens(text):
    """精确计算文本的 token 数量"""
    if not text:
        return 0
    return len(_get_encoder().encode(text))


class MemoryRetriever:
    """
    多级记忆检索器。

    初始化时绑定 MemoryStore 和共现矩阵（可选），之后通过 retrieve() 检索。
    共现矩阵和 vocab_dict 可以后续通过 set_association_matrix() 设置或更新。

    用法：
        retriever = MemoryRetriever(store)
        retriever.set_association_matrix(ppmi_matrix, vocab_dict)

        # 日常对话：快速
        result = retriever.retrieve(keywords=["关税"], depth="fast")

        # 需要联想：标准
        result = retriever.retrieve(keywords=["关税","贸易"], depth="standard")

        # 复杂推理：深度
        result = retriever.retrieve(keywords=["特朗普","关税","A股"], depth="deep")
    """

    def __init__(self, store, ppmi_matrix=None, vocab_dict=None, cooccurrence_matrix=None):
        """
        参数:
            store: MemoryStore — 记忆存储实例
            ppmi_matrix: scipy csr_matrix | None — PPMI 矩阵（standard/deep 模式需要）
            vocab_dict: dict | None — 词到索引的映射
            cooccurrence_matrix: scipy sparse matrix | None — 原始共现矩阵（deep 模式 fallback）
        """
        self.store = store
        self.ppmi_matrix = ppmi_matrix
        self.vocab_dict = vocab_dict
        self.cooccurrence_matrix = cooccurrence_matrix
        # 缓存反向映射 idx→word，避免每次 _expand_keywords 都重建
        self._idx_to_word = {v: k for k, v in vocab_dict.items()} if vocab_dict else {}

    def set_association_matrix(self, ppmi_matrix, vocab_dict, cooccurrence_matrix=None):
        """设置或更新 PPMI 关联矩阵"""
        self.ppmi_matrix = ppmi_matrix
        self.vocab_dict = vocab_dict
        # 更新反向映射缓存
        self._idx_to_word = {v: k for k, v in vocab_dict.items()} if vocab_dict else {}
        if cooccurrence_matrix is not None:
            self.cooccurrence_matrix = cooccurrence_matrix

    # ========================
    # 主入口
    # ========================

    def retrieve(self,
                 user_input=None,
                 keywords=None,
                 depth="standard",
                 token_budget=1000,
                 time_range=None,
                 time_recent=None,
                 time_decay=0.1,
                 source_filter=None,
                 top_n_expand=10,
                 top_n_entries=10,
                 chain_fuzzy=False,
                 long_threshold=80,
                 core_ratio=0.2,
                 important_ratio=0.4):
        """
        多级记忆检索。

        参数:
            user_input:    str | None — 原始用户输入文本（自动提取关键词）
            keywords:      list[str] | None — AI 直接给的关键词（优先级高于 user_input）
            depth:         str — "fast" / "standard" / "deep"
            token_budget:  int — prompt 注入的最大 token 数
            time_range:    tuple(start_ts, end_ts) | None — 精确时间范围过滤
            time_recent:   float | None — 最近 N 小时
            time_decay:    float — 时间衰减系数 λ（反比函数 1/(1+λ*days)）
            source_filter: str | None — 只搜某个智能体的记忆
            top_n_expand:  int — 关联扩展时取 top-N 关联词
            top_n_entries: int — 最终返回最多几条记忆条目
            chain_fuzzy:   bool — 隐藏链推理词是否也做模糊匹配（默认 False）
            long_threshold: int — 长文本模式触发阈值（字符数），默认 80
            core_ratio:    float — 核心词占提取总词数的比例，默认 0.2
            important_ratio: float — 重要词占提取总词数的比例，默认 0.4

        返回:
            dict — {
                "prompt_text":       str,       # 拼好的注入文本（≤token_budget）
                "matched_entries":   list[dict], # 命中条目详情
                "expanded_keywords": list[str],  # 关联扩展的词（standard/deep）
                "chain":             dict|None,  # 推理链结果（deep 才有）
                "search_stats":      dict,       # 搜索统计
            }
        """
        t_start = time.time()
        stats = {
            "depth_used": depth,
            "total_entries_scanned": len(self.store),
            "time_filtered": 0,
            "exact_hits": 0,
            "fuzzy_hits": 0,
            "assoc_hits": 0,
            "long_text_mode": False,
            "time_ms": 0,
        }

        # === Step 0: 确定查询关键词（含长文本智能分级） ===

        # AI 直接给的关键词（已经过 AI 判断，视为核心）
        ai_keywords = list(keywords) if keywords else []

        # 判断是否长文本模式
        is_long = (user_input and isinstance(user_input, str)
                   and len(user_input) > long_threshold)

        if is_long:
            # ---- 长文本模式：TF-IDF 提取 + 分级 ----
            stats["long_text_mode"] = True
            weighted_kws = extract_keywords_weighted(
                user_input, top_k=25, long_threshold=long_threshold)

            # 去掉 AI 已给的关键词（避免重复）
            ai_set = set(ai_keywords)
            weighted_kws = [(w, s) for w, s in weighted_kws if w not in ai_set]

            total = len(weighted_kws)
            core_count = max(3, int(total * core_ratio))
            important_count = max(3, int(total * important_ratio))

            # 分三级（weighted_kws 已按权重降序）
            core_words = [w for w, _ in weighted_kws[:core_count]]
            important_words = [w for w, _ in weighted_kws[core_count:core_count + important_count]]
            supplement_words = [w for w, _ in weighted_kws[core_count + important_count:]]

            # AI 给的关键词并入核心词（它们是 AI 已判断过的重点）
            core_keywords = ai_keywords + core_words

            # 各 Tier 使用的关键词集合
            tier0_keywords = core_keywords + important_words    # 精确: 核心 + 重要
            tier1_keywords = core_keywords                      # 模糊: 仅核心
            tier2_keywords = core_keywords                      # 扩展: 仅核心
            tier3_keywords = core_keywords                      # 链:   仅核心

            stats["core_keywords"] = core_keywords
            stats["important_keywords"] = important_words
            stats["supplement_keywords"] = supplement_words

        else:
            # ---- 短文本模式：原有逻辑不变 ----
            if ai_keywords and user_input:
                # 两者都给了，合并去重
                auto_kw = extract_keywords_jieba(user_input)
                merged = list(ai_keywords)
                for w in auto_kw:
                    if w not in merged:
                        merged.append(w)
                query_keywords = merged
            elif ai_keywords:
                query_keywords = ai_keywords
            elif user_input:
                query_keywords = extract_keywords_jieba(user_input)
            else:
                query_keywords = []

            if not query_keywords:
                return self._empty_result(stats, t_start)

            # 短文本：所有 Tier 用同一组关键词
            tier0_keywords = query_keywords
            tier1_keywords = query_keywords
            tier2_keywords = query_keywords
            tier3_keywords = query_keywords
            core_keywords = query_keywords
            supplement_words = []

        if not tier0_keywords:
            return self._empty_result(stats, t_start)

        # === Step 1: 时间范围预过滤（确定候选池） ===
        if time_range or time_recent:
            all_ids = list(self.store.entries.keys())
            # 如果有 source_filter，先过滤来源
            if source_filter:
                all_ids = [eid for eid in all_ids
                           if self.store.entries[eid].get("source") == source_filter]
            candidate_pool = set(self.store.filter_by_time(
                all_ids, time_range=time_range, time_recent=time_recent))
            stats["time_filtered"] = len(all_ids) - len(candidate_pool)
        else:
            candidate_pool = None   # None 表示不限制

        # === Step 2: Tier 0 精确匹配 ===
        exact_hits = self.store.search_exact(tier0_keywords)
        if candidate_pool is not None:
            exact_hits = {eid: n for eid, n in exact_hits.items()
                          if eid in candidate_pool}
        if source_filter and candidate_pool is None:
            exact_hits = {eid: n for eid, n in exact_hits.items()
                          if self.store.entries.get(eid, {}).get("source") == source_filter}
        stats["exact_hits"] = len(exact_hits)

        # === Step 3: Tier 1 模糊匹配 ===
        fuzzy_hits = self.store.search_fuzzy(tier1_keywords)
        if candidate_pool is not None:
            fuzzy_hits = {eid: s for eid, s in fuzzy_hits.items()
                          if eid in candidate_pool}
        if source_filter and candidate_pool is None:
            fuzzy_hits = {eid: s for eid, s in fuzzy_hits.items()
                          if self.store.entries.get(eid, {}).get("source") == source_filter}
        stats["fuzzy_hits"] = len(fuzzy_hits)

        # === Step 4: Tier 2 关联扩展（standard / deep）===
        expanded_keywords = []
        assoc_hits = {}

        if depth in ("standard", "deep") and self.ppmi_matrix is not None and self.vocab_dict is not None:
            expanded_keywords = self._expand_keywords(tier2_keywords, top_n=top_n_expand)
            if expanded_keywords:
                assoc_exact = self.store.search_exact(expanded_keywords)
                assoc_fuzzy = self.store.search_fuzzy(expanded_keywords)
                # 合并，关联扩展的命中权重减半
                for eid, n in assoc_exact.items():
                    if candidate_pool is not None and eid not in candidate_pool:
                        continue
                    assoc_hits[eid] = assoc_hits.get(eid, 0) + n * 0.5
                for eid, s in assoc_fuzzy.items():
                    if candidate_pool is not None and eid not in candidate_pool:
                        continue
                    assoc_hits[eid] = max(assoc_hits.get(eid, 0), s * 0.3)
                stats["assoc_hits"] = len(assoc_hits)

        # === Step 5: Tier 3 隐藏链推理（deep）===
        chain_result = None
        if depth == "deep" and self.ppmi_matrix is not None and self.vocab_dict is not None:
            if len(tier3_keywords) >= 2:
                chain_result = discover_hidden_chain(
                    self.ppmi_matrix, self.vocab_dict, tier3_keywords,
                    top_k_candidates=50, alpha=0.85, min_edge_weight=0.1, top_n=10,
                    fallback_matrix=self.cooccurrence_matrix
                )
                # 链上的隐藏词也去查记忆
                if chain_result and chain_result.get("hidden_words"):
                    chain_words = [hw["word"] for hw in chain_result["hidden_words"][:5]]
                    chain_hits = self.store.search_exact(chain_words)
                    for eid, n in chain_hits.items():
                        if candidate_pool is not None and eid not in candidate_pool:
                            continue
                        assoc_hits[eid] = assoc_hits.get(eid, 0) + n * 0.4
                    # 可选：隐藏链词也做模糊匹配
                    if chain_fuzzy:
                        chain_fuzzy_hits = self.store.search_fuzzy(chain_words)
                        for eid, s in chain_fuzzy_hits.items():
                            if candidate_pool is not None and eid not in candidate_pool:
                                continue
                            assoc_hits[eid] = max(assoc_hits.get(eid, 0), s * 0.2)

        # === Step 6: 综合评分 ===
        all_entry_ids = set(exact_hits.keys()) | set(fuzzy_hits.keys()) | set(assoc_hits.keys())
        if not all_entry_ids:
            return self._empty_result(stats, t_start, expanded_keywords, chain_result)

        now = time.time()
        scored_entries = []
        for eid in all_entry_ids:
            entry = self.store.get(eid)
            if entry is None:
                continue

            # 关联维度
            exact_score = exact_hits.get(eid, 0)
            fuzzy_score = fuzzy_hits.get(eid, 0)
            assoc_score = assoc_hits.get(eid, 0)
            relevance = exact_score * 1.0 + fuzzy_score * 0.6 + assoc_score * 0.4

            # 长文本模式：补充词二次验证（加分不扣分）
            if supplement_words:
                entry_kws = set(entry.get("keywords") or [])
                entry_text = (entry.get("summary") or "") + (entry.get("body") or "")
                supplement_match = 0
                for sw in supplement_words:
                    if sw in entry_kws or sw in entry_text:
                        supplement_match += 1
                # 验证系数 1.0 ~ 1.5（命中全部补充词时为 1.5）
                verify_bonus = 1.0 + 0.5 * (supplement_match / len(supplement_words))
                relevance *= verify_bonus

            # 时间维度（反比衰减，importance 感知）
            entry_importance = entry.get("importance", 3)
            tw = MemoryStore.compute_time_weight(
                entry["timestamp"], decay_lambda=time_decay, now=now,
                importance=entry_importance)

            # 重要度因子: importance=3 时为 1.0（中性），5 时为 1.67，1 时为 0.33
            importance_factor = entry_importance / 3.0

            final_score = relevance * tw * importance_factor

            age_hours = (now - entry["timestamp"]) / 3600.0

            scored_entries.append({
                "entry_id": eid,
                "entry": entry,
                "relevance_score": round(relevance, 4),
                "time_weight": round(tw, 4),
                "final_score": round(final_score, 4),
                "age_hours": round(age_hours, 2),
            })

        # 按综合分降序
        scored_entries.sort(key=lambda x: x["final_score"], reverse=True)
        scored_entries = scored_entries[:top_n_entries]

        # === Step 7: 分层加载（token 预算内）===
        prompt_text, loaded_info = self._layered_load(scored_entries, token_budget,
                                                      expanded_keywords, chain_result)

        stats["time_ms"] = round((time.time() - t_start) * 1000, 2)

        # 构建返回的 matched_entries（不含原始 entry 全文，避免冗余）
        matched_output = []
        for se in scored_entries:
            entry = se["entry"]
            matched_output.append({
                "id": se["entry_id"],
                "entry_id": se["entry_id"],
                "topic": entry.get("topic"),
                "keywords": entry.get("keywords"),
                "relevance_score": se["relevance_score"],
                "time_weight": se["time_weight"],
                "importance": entry.get("importance", 3),
                "final_score": se["final_score"],
                "timestamp": entry.get("timestamp"),
                "age_hours": se["age_hours"],
                "loaded_layers": loaded_info.get(se["entry_id"], []),
            })

        return {
            "prompt_text": prompt_text,
            "matched_entries": matched_output,
            "expanded_keywords": expanded_keywords,
            "chain": chain_result,
            "search_stats": stats,
        }

    # ========================
    # 关联扩展
    # ========================

    def _expand_keywords(self, keywords, top_n=10):
        """
        通过 PPMI 矩阵扩展关键词。

        对每个输入词，找到 PPMI 最高的 top_n 个关联词（排除输入词本身）。
        多个输入词的关联词按 PPMI 分数汇总后取 top_n。
        """
        if self.ppmi_matrix is None or self.vocab_dict is None:
            return []

        input_set = set(keywords)
        assoc_scores = {}

        for word in keywords:
            if word not in self.vocab_dict:
                continue
            idx = self.vocab_dict[word]
            row = self.ppmi_matrix[idx, :].toarray().flatten()

            # 找非零项
            nonzero = np.where(row > 0)[0]

            for nz_idx in nonzero:
                w = self._idx_to_word.get(nz_idx)
                if w is None or w in input_set:
                    continue
                score = row[nz_idx]
                assoc_scores[w] = assoc_scores.get(w, 0) + score

        # 按分数降序取 top_n
        sorted_assoc = sorted(assoc_scores.items(), key=lambda x: x[1], reverse=True)
        return [w for w, s in sorted_assoc[:top_n]]

    # ========================
    # 分层加载
    # ========================

    def _layered_load(self, scored_entries, token_budget, expanded_keywords=None,
                      chain_result=None):
        """
        在 token 预算内分层加载记忆条目。

        加载优先级：topic → keywords → summary → body（截断）
        缺失字段自动跳过。
        最后附加关联词提示（如果还有预算）。

        返回:
            tuple(str, dict) — (拼装的 prompt 文本, {entry_id: [已加载的层名]})
        """
        parts = []
        used_tokens = 0
        loaded_info = {}    # entry_id → [layer_names]

        # 头部标记
        header = "[记忆上下文]\n"
        header_tokens = count_tokens(header)
        if header_tokens < token_budget:
            parts.append(header)
            used_tokens += header_tokens

        for i, se in enumerate(scored_entries):
            entry = se["entry"]
            eid = se["entry_id"]
            loaded_info[eid] = []
            remaining = token_budget - used_tokens

            if remaining <= 20:     # 剩余空间太少，停止
                break

            entry_parts = []
            entry_prefix = f"\n--- 记忆 #{i+1} "
            if entry.get("source"):
                entry_prefix += f"[来源:{entry['source']}] "
            # 绝对时间
            import time as _time
            ts = entry.get("timestamp")
            if ts:
                abs_time = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(ts))
                entry_prefix += f"[{abs_time}] "

            # 相对时间
            age_h = se["age_hours"]
            if age_h < 1:
                entry_prefix += f"[{age_h*60:.0f}分钟前]"
            elif age_h < 24:
                entry_prefix += f"[{age_h:.1f}小时前]"
            else:
                entry_prefix += f"[{age_h/24:.1f}天前]"
            entry_prefix += " ---\n"

            prefix_tokens = count_tokens(entry_prefix)
            if prefix_tokens >= remaining:
                break

            entry_parts.append(entry_prefix)
            used_tokens += prefix_tokens
            remaining -= prefix_tokens

            # Layer 1: topic
            if entry.get("topic"):
                text = f"主题: {entry['topic']}\n"
                t = count_tokens(text)
                if t <= remaining:
                    entry_parts.append(text)
                    used_tokens += t
                    remaining -= t
                    loaded_info[eid].append("topic")

            # Layer 2: keywords
            if entry.get("keywords"):
                text = f"关键词: {', '.join(entry['keywords'])}\n"
                t = count_tokens(text)
                if t <= remaining:
                    entry_parts.append(text)
                    used_tokens += t
                    remaining -= t
                    loaded_info[eid].append("keywords")

            # Layer 3: summary
            if entry.get("summary"):
                text = f"摘要: {entry['summary']}\n"
                t = count_tokens(text)
                if t <= remaining:
                    entry_parts.append(text)
                    used_tokens += t
                    remaining -= t
                    loaded_info[eid].append("summary")
                elif remaining > 30:
                    # 预算不够加全部 summary，截断
                    truncated = self._truncate_to_tokens(
                        f"摘要: {entry['summary']}", remaining - 5)
                    if truncated:
                        text = truncated + "...\n"
                        t = count_tokens(text)
                        entry_parts.append(text)
                        used_tokens += t
                        remaining -= t
                        loaded_info[eid].append("summary(截断)")

            # Layer 4: body（只对排名第一的条目尝试加载，且需要足够预算）
            if i == 0 and entry.get("body") and remaining > 50:
                body_text = f"详情: {entry['body']}\n"
                t = count_tokens(body_text)
                if t <= remaining:
                    entry_parts.append(body_text)
                    used_tokens += t
                    remaining -= t
                    loaded_info[eid].append("body")
                elif remaining > 80:
                    truncated = self._truncate_to_tokens(
                        f"详情: {entry['body']}", remaining - 5)
                    if truncated:
                        text = truncated + "...\n"
                        t = count_tokens(text)
                        entry_parts.append(text)
                        used_tokens += t
                        remaining -= t
                        loaded_info[eid].append("body(截断)")

            parts.extend(entry_parts)

        # 附加关联词提示（如有预算）
        if expanded_keywords and (token_budget - used_tokens) > 20:
            hint = f"\n[关联概念] {', '.join(expanded_keywords[:8])}\n"
            t = count_tokens(hint)
            if t <= token_budget - used_tokens:
                parts.append(hint)
                used_tokens += t

        # 附加推理链提示（deep 模式，如有预算）
        if chain_result and (token_budget - used_tokens) > 30:
            chain_parts = []

            # 隐藏关联词
            hidden = chain_result.get("hidden_words", [])
            if hidden:
                hw_list = [hw["word"] for hw in hidden[:6]]
                chain_parts.append(f"[隐藏关联] {', '.join(hw_list)}")

            # 推理路径
            valid_chains = chain_result.get("chains", [])
            if valid_chains:
                chain_parts.append("[推理路径]")
                for c in valid_chains[:3]:  # 最多展示 3 条路径
                    path = c.get("path")
                    if path:
                        chain_parts.append(
                            f"  {c['from']} → {' → '.join(path[1:-1])} → {c['to']}"
                            f" (关联强度:{c['total_weight']:.3f})"
                        )

            if chain_parts:
                chain_text = "\n" + "\n".join(chain_parts) + "\n"
                t = count_tokens(chain_text)
                if t <= token_budget - used_tokens:
                    parts.append(chain_text)
                    used_tokens += t

        prompt_text = "".join(parts)
        return prompt_text, loaded_info

    def _truncate_to_tokens(self, text, max_tokens):
        """将文本截断到不超过 max_tokens"""
        encoder = _get_encoder()
        tokens = encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return encoder.decode(truncated_tokens)

    # ========================
    # 工具
    # ========================

    def _empty_result(self, stats, t_start, expanded_keywords=None, chain=None):
        stats["time_ms"] = round((time.time() - t_start) * 1000, 2)
        return {
            "prompt_text": "",
            "matched_entries": [],
            "expanded_keywords": expanded_keywords or [],
            "chain": chain,
            "search_stats": stats,
        }
