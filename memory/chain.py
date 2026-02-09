# -*- coding: utf-8 -*-
"""
隐藏链推理模块（PPR + 最短路径）

核心场景：
    给定稀疏锚点词 [a, e]，在共现图中发现隐藏的 b, c, d, f，
    重建完整的概念关联链。

算法流程：
    1. PPMI 稀疏矩阵 → NetworkX 加权图（边权 = PPMI，距离 = 1/PPMI）
    2. Personalized PageRank：以锚点词为种子，找 top-K 候选隐藏词
    3. 在全图上求锚点两两 Dijkstra 最短路径 → 提取中间节点 + 投票
    4. PPR 分数 × 路径投票 综合排序（路径中间节点额外加分）→ 输出隐藏词 + 链结构
"""

import numpy as np
import networkx as nx
from collections import defaultdict


def _build_networkx_graph(ppmi_matrix, idx_to_word, min_weight=0.1):
    """
    将 PPMI 稀疏矩阵转换为 NetworkX 加权无向图。

    边权属性：
        - weight: PPMI 值（越大关联越强）
        - distance: 1 / (PPMI + ε)（用于最短路径，越小越近）

    参数:
        ppmi_matrix: scipy csr_matrix — PPMI 矩阵
        idx_to_word: dict — 索引到词的映射
        min_weight: float — 最小 PPMI 阈值，低于此值的边不加入图
    返回:
        nx.Graph — 加权无向图
    """
    G = nx.Graph()
    coo = ppmi_matrix.tocoo()

    for r, c, v in zip(coo.row, coo.col, coo.data):
        if r >= c:          # 无向图只加一条边，跳过对角线和重复
            continue
        if v < min_weight:
            continue

        word_r = idx_to_word.get(r)
        word_c = idx_to_word.get(c)
        if word_r is None or word_c is None:
            continue

        G.add_edge(word_r, word_c,
                   weight=float(v),
                   distance=1.0 / (float(v) + 1e-8))

    return G


def _personalized_pagerank(G, anchor_words, alpha=0.85, top_k=50):
    """
    在图上执行 Personalized PageRank。

    种子节点为所有锚点词，均匀分配初始概率。
    返回 top-K 个非锚点节点及其 PPR 分数。

    参数:
        G: nx.Graph — 加权图
        anchor_words: list[str] — 锚点词列表
        alpha: float — 阻尼因子（0.85 为经典值）
        top_k: int — 返回候选词数量
    返回:
        list[tuple(str, float)] — [(word, ppr_score), ...] 按分数降序
    """
    # 构建 personalization 向量：锚点词均匀分配
    anchor_set = set(anchor_words) & set(G.nodes())
    if not anchor_set:
        return []

    personalization = {}
    share = 1.0 / len(anchor_set)
    for node in G.nodes():
        personalization[node] = share if node in anchor_set else 0.0

    # 执行 PPR
    ppr_scores = nx.pagerank(G, alpha=alpha,
                             personalization=personalization,
                             weight='weight',
                             max_iter=100, tol=1e-6)

    # 排除锚点词，按分数降序取 top-K
    candidates = [(word, score) for word, score in ppr_scores.items()
                  if word not in anchor_set]
    candidates.sort(key=lambda x: x[1], reverse=True)

    return candidates[:top_k]


def _check_anchors_connected(G, anchor_words):
    """
    检查所有锚点词是否在同一个连通分量中。

    参数:
        G: nx.Graph — 加权图
        anchor_words: list[str] — 锚点词列表
    返回:
        bool — 是否全部连通
    """
    anchors_in = [w for w in anchor_words if w in G.nodes()]
    if len(anchors_in) < 2:
        return False
    # 检查所有锚点是否两两连通
    first = anchors_in[0]
    comp = nx.node_connected_component(G, first)
    return all(w in comp for w in anchors_in[1:])


def _find_shortest_paths(G, anchor_words):
    """
    在图上求锚点词两两之间的最短路径（Dijkstra，基于 distance 属性）。

    返回所有路径及中间节点统计。

    参数:
        G: nx.Graph — 加权图
        anchor_words: list[str] — 锚点词列表
    返回:
        tuple(list[dict], dict) —
            paths: [{"from": a, "to": b, "path": [...], "total_weight": float}, ...]
            intermediate_counts: {word: 出现在多少条路径中}
    """
    anchors_in_graph = [w for w in anchor_words if w in G.nodes()]
    paths = []
    intermediate_counts = defaultdict(int)

    for i in range(len(anchors_in_graph)):
        for j in range(i + 1, len(anchors_in_graph)):
            src = anchors_in_graph[i]
            tgt = anchors_in_graph[j]

            try:
                path = nx.dijkstra_path(G, src, tgt, weight='distance')
                # 计算路径的总 PPMI 权重
                total_w = 0
                for k in range(len(path) - 1):
                    edge_data = G.get_edge_data(path[k], path[k + 1])
                    total_w += edge_data.get('weight', 0) if edge_data else 0

                paths.append({
                    "from": src,
                    "to": tgt,
                    "path": path,
                    "total_weight": round(total_w, 4),
                    "hops": len(path) - 1,
                })

                # 统计中间节点（排除首尾锚点）
                anchor_set = set(anchor_words)
                for node in path[1:-1]:
                    if node not in anchor_set:
                        intermediate_counts[node] += 1

            except nx.NetworkXNoPath:
                paths.append({
                    "from": src,
                    "to": tgt,
                    "path": None,
                    "total_weight": 0,
                    "hops": -1,
                })

    return paths, dict(intermediate_counts)


def discover_hidden_chain(ppmi_matrix, vocab_dict, anchor_words,
                          top_k_candidates=50, alpha=0.85,
                          min_edge_weight=0.1, top_n=10,
                          auto_reduce_weight=True,
                          fallback_matrix=None):
    """
    隐藏链推理：给定稀疏锚点词，发现隐藏的中间关联词和链结构。

    算法：
        1. PPMI 矩阵 → NetworkX 加权图
        2. Personalized PageRank 找候选隐藏词
        3. 在全图上求锚点两两最短路径（非子图，避免稀疏丢路径）
        4. 综合 PPR 分数 × 路径投票 排序（路径中间节点额外加分）

    参数:
        ppmi_matrix: scipy csr_matrix — PPMI 矩阵
        vocab_dict: dict — 词到索引的映射
        anchor_words: list[str] — 锚点词列表（至少 2 个）
        top_k_candidates: int — PPR 候选词数量
        alpha: float — PPR 阻尼因子
        min_edge_weight: float — 图的最小边权（PPMI 低于此值不建边）
        top_n: int — 最终返回的隐藏词数量
        auto_reduce_weight: bool — 锚点不连通时自动降低边权阈值重建图
        fallback_matrix: scipy sparse matrix | None — 备用矩阵（如原始共现矩阵），
            当 PPMI 图中锚点不连通时用此矩阵补充弱关联边

    返回:
        dict — {
            "hidden_words": [
                {"word": str, "ppr_score": float, "path_count": int,
                 "combined_score": float},
                ...
            ],
            "chains": [
                {"from": str, "to": str, "path": list, "total_weight": float, "hops": int},
                ...
            ],
            "subgraph_nodes": int,
            "subgraph_edges": int,
            "anchors_found": list[str],
            "anchors_missing": list[str],
        }
    """
    idx_to_word = {v: k for k, v in vocab_dict.items()}

    # --- Step 1: 构建全局图 ---
    G_full = _build_networkx_graph(ppmi_matrix, idx_to_word,
                                   min_weight=min_edge_weight)

    # 检查锚点词是否在图中
    anchors_found = [w for w in anchor_words if w in G_full.nodes()]
    anchors_missing = [w for w in anchor_words if w not in G_full.nodes()]

    if len(anchors_found) < 2:
        return {
            "hidden_words": [],
            "chains": [],
            "subgraph_nodes": 0,
            "subgraph_edges": 0,
            "anchors_found": anchors_found,
            "anchors_missing": anchors_missing,
            "error": f"图中至少需要 2 个锚点词，仅找到 {len(anchors_found)} 个",
        }

    # --- Step 1.5: 连通性检查 + 自动降权 ---
    # 如果锚点词不在同一个连通分量，尝试补救：
    #   a) 降低 min_edge_weight 阈值重建 PPMI 图
    #   b) 如果仍不连通且有 fallback_matrix，用原始共现矩阵补边
    if auto_reduce_weight and len(anchors_found) >= 2:
        _all_connected = _check_anchors_connected(G_full, anchors_found)
        if not _all_connected:
            # 尝试降低阈值
            for try_weight in [min_edge_weight * 0.5, 0.05, 0.01, 0.001, 0.0]:
                G_try = _build_networkx_graph(ppmi_matrix, idx_to_word,
                                              min_weight=try_weight)
                if _check_anchors_connected(G_try, anchors_found):
                    G_full = G_try
                    _all_connected = True
                    break

            # 如果 PPMI 图降到 0 仍不连通，用 fallback_matrix 补边
            if not _all_connected and fallback_matrix is not None:
                G_fallback = _build_networkx_graph(fallback_matrix, idx_to_word,
                                                   min_weight=0.5)
                # 将 fallback 图中的边补充到 G_full（只加不覆盖）
                for u, v, data in G_fallback.edges(data=True):
                    if not G_full.has_edge(u, v):
                        # fallback 边权重降低，标记为弱关联
                        G_full.add_edge(u, v,
                                        weight=data['weight'] * 0.1,
                                        distance=data['distance'] * 10)
                # 再次重建 anchors_found（fallback 可能加入了新节点）
                anchors_found = [w for w in anchor_words if w in G_full.nodes()]
                anchors_missing = [w for w in anchor_words if w not in G_full.nodes()]

    # --- Step 2: Personalized PageRank ---
    ppr_candidates = _personalized_pagerank(G_full, anchors_found,
                                            alpha=alpha,
                                            top_k=top_k_candidates)

    if not ppr_candidates:
        return {
            "hidden_words": [],
            "chains": [],
            "subgraph_nodes": len(anchors_found),
            "subgraph_edges": 0,
            "anchors_found": anchors_found,
            "anchors_missing": anchors_missing,
            "error": "PPR 未找到候选词",
        }

    # PPR 分数映射
    ppr_score_map = {word: score for word, score in ppr_candidates}

    # --- Step 3: 在全图上求锚点两两最短路径 ---
    # 注意：之前在候选子图上跑最短路径，子图太稀疏导致找不到路径。
    # 改为在全图上跑，PPR 分数只用于排序加分，不限制搜索范围。
    chains, intermediate_counts = _find_shortest_paths(G_full, anchors_found)

    # 记录候选子图统计（用于返回信息，不影响逻辑）
    subgraph_nodes = set(anchors_found) | set(ppr_score_map.keys())
    n_sub_nodes = len(subgraph_nodes & set(G_full.nodes()))
    n_sub_edges = G_full.subgraph(subgraph_nodes).number_of_edges()

    # --- Step 4: 综合评分 ---
    # combined_score = ppr_score × (1 + path_count) + path_only_bonus
    # - PPR 高分节点：即使不在路径上也有基础分（ppr_score × 1）
    # - 路径中间节点：path_count 越高加成越大
    # - 仅在路径上但不在 PPR 候选中的节点：给一个 path_only_bonus
    PATH_ONLY_BONUS = 0.001   # 仅通过路径发现的节点基础分

    # 收集所有需要评分的词：PPR 候选 + 路径中间节点
    all_scored_words = set(ppr_score_map.keys()) | set(intermediate_counts.keys())

    hidden_words = []
    for word in all_scored_words:
        ppr_score = ppr_score_map.get(word, 0.0)
        path_count = intermediate_counts.get(word, 0)

        if ppr_score > 0:
            combined = ppr_score * (1.0 + path_count)
        else:
            # 仅通过路径发现的节点
            combined = PATH_ONLY_BONUS * path_count

        hidden_words.append({
            "word": word,
            "ppr_score": round(ppr_score, 6),
            "path_count": path_count,
            "combined_score": round(combined, 6),
        })

    # 按综合分排序
    hidden_words.sort(key=lambda x: x["combined_score"], reverse=True)
    hidden_words = hidden_words[:top_n]

    # 过滤掉没有有效路径的 chains
    valid_chains = [c for c in chains if c["path"] is not None]

    return {
        "hidden_words": hidden_words,
        "chains": valid_chains,
        "subgraph_nodes": n_sub_nodes,
        "subgraph_edges": n_sub_edges,
        "anchors_found": anchors_found,
        "anchors_missing": anchors_missing,
    }


def discover_hidden_chain_with_evidence(ppmi_matrix, vocab_dict, anchor_words,
                                        memory_store=None, top_k_candidates=50,
                                        alpha=0.85, min_edge_weight=0.1,
                                        top_n=10, fallback_matrix=None):
    """
    隐藏链推理 + 文本证据关联。

    在 discover_hidden_chain 的基础上，为每个隐藏词从 MemoryStore 中
    检索关联的文本片段（摘要层），作为 AI 判断的依据。

    参数:
        memory_store: MemoryStore | None — 记忆存储实例
        fallback_matrix: 备用矩阵（同 discover_hidden_chain）
        其余参数同 discover_hidden_chain

    返回:
        dict — 在 discover_hidden_chain 返回值基础上增加:
            "evidence": [
                {"word": str, "entry_id": str, "topic": str, "summary": str},
                ...
            ]
    """
    result = discover_hidden_chain(
        ppmi_matrix, vocab_dict, anchor_words,
        top_k_candidates=top_k_candidates,
        alpha=alpha, min_edge_weight=min_edge_weight,
        top_n=top_n, fallback_matrix=fallback_matrix
    )

    # 如果没有 MemoryStore 或没找到隐藏词，直接返回
    if memory_store is None or not result.get("hidden_words"):
        result["evidence"] = []
        return result

    # 为每个隐藏词查询关联的记忆条目
    evidence = []
    for hw in result["hidden_words"]:
        word = hw["word"]
        # 精确搜索 + 模糊搜索
        exact_hits = memory_store.search_exact([word])
        fuzzy_hits = memory_store.search_fuzzy([word])

        # 合并，取最佳命中
        best_eid = None
        if exact_hits:
            best_eid = list(exact_hits.keys())[0]
        elif fuzzy_hits:
            best_eid = list(fuzzy_hits.keys())[0]

        if best_eid:
            entry = memory_store.get(best_eid)
            if entry:
                evidence.append({
                    "word": word,
                    "entry_id": best_eid,
                    "topic": entry.get("topic"),
                    "summary": entry.get("summary"),
                })

    result["evidence"] = evidence
    return result
