# -*- coding: utf-8 -*-
"""
Agent 长期关联记忆模块 (SQLite 版)

主入口:
    from memory import AgentMemory

    # 纯内存模式（向后兼容）
    am = AgentMemory()

    # SQLite 持久化模式（推荐）
    am = AgentMemory(data_dir="./memory_data")

三种记忆类型:
    knowledge — 事实、概念、稳定知识（语义记忆）
    experience — 经验教训、模式总结（经验记忆）
    log — 活动事件、发生了什么（事件记忆）

三大脑启发机制:
    提取强化 — query 命中自动递增 access_count
    重要度分级 — importance 1~5 影响衰减和排名
    记忆再巩固 — supersedes 纠正旧记忆

核心组件:
- agent_memory: AgentMemory 统一 API（推荐使用）
- cooccurrence: 增量共现矩阵管理（支持文档全文 + 关键词列表双通道学习）
- probability: PPMI / 条件概率矩阵计算
- decay: 时间衰减管理
- inference: 多跳推理搜索（2跳 + Beam Search v3.1）
- storage: SQLite 记忆条目存储 + keyword_index 倒排索引 + 模糊搜索
- chain: 隐藏链推理（PPR + 最短路径）
- retriever: 多级检索器（fast/standard/deep）+ 分层加载 + token预算
"""

# 统一 API（推荐入口）
from .agent_memory import AgentMemory

# 底层模块（高级用户直接使用）
from .cooccurrence import IncrementalCooccurrence
from .probability import compute_ppmi_matrix, compute_conditional_prob_matrix
from .decay import DecayManager
from .inference import find_top_inference_paths, find_group_inference_paths, beam_search_inference
from .storage import MemoryStore, extract_keywords_jieba, extract_nouns_jieba, extract_keywords_weighted, parse_time_expression
from .chain import discover_hidden_chain, discover_hidden_chain_with_evidence
from .retriever import MemoryRetriever, count_tokens
