# -*- coding: utf-8 -*-
"""
多跳推理搜索模块
- find_top_inference_paths: 单词 2 跳推理（兼容原方案）
- find_group_inference_paths: 多词联合 2 跳推理（兼容原方案）
- beam_search_inference: N 跳自适应 Beam Search 推理（新增）
"""
import numpy as np
from heapq import nlargest


def _merge_substring_targets(results, top_n):
    """
    子串合并后处理：将推理结果中的子串碎片合并到对应的长词上。

    问题场景：
        jieba 分词可能同时产生 "特斯", "斯拉", "特斯拉"，
        推理时这些碎片各自独立积累得分，稀释了长词的真实排名。

    策略：
        1. 按目标词长度降序排列
        2. 如果短词是某长词的子串，将短词的得分合并到长词上
        3. 合并后的路径解释保留得分最高的前 3 条
    """
    if not results or isinstance(results, str):
        return results

    # 按目标词长度降序 + 置信度降序排列
    results.sort(key=lambda x: (len(x['target']), x['confidence']), reverse=True)

    merged = []  # 保留的结果
    merged_words = []  # 对应的目标词列表

    for item in results:
        word = item['target']
        # 检查是否是已保留某长词的子串
        merged_into = None
        for i, kept_word in enumerate(merged_words):
            if word in kept_word and word != kept_word:
                merged_into = i
                break

        if merged_into is not None:
            # 将得分合并到长词上
            merged[merged_into]['confidence'] += item['confidence']
            merged[merged_into]['confidence'] = round(merged[merged_into]['confidence'], 6)
            # 合并路径解释
            merged[merged_into]['why'].extend(item.get('why', []))
            # 按得分重新排序，只保留前 3 条
            sort_key = 'score'
            merged[merged_into]['why'] = sorted(
                merged[merged_into]['why'], key=lambda x: x.get(sort_key, 0), reverse=True
            )[:3]
        else:
            merged.append(item)
            merged_words.append(word)

    # 按合并后的置信度重新排序
    merged.sort(key=lambda x: x['confidence'], reverse=True)
    return merged[:top_n]


def find_top_inference_paths(prob_matrix, vocab_dict, start_word, top_n=5):
    """
    单词 2 跳推理路径搜索：A -> B -> C

    参数:
        prob_matrix: 概率/PPMI矩阵
        vocab_dict: dict - 词到索引的映射
        start_word: str - 起始词
        top_n: int - 返回前 N 条路径
    """
    if start_word not in vocab_dict:
        return f"词 '{start_word}' 不在词库中"

    idx_a = vocab_dict[start_word]
    idx_to_word = {v: k for k, v in vocab_dict.items()}

    row_a = prob_matrix[idx_a, :].toarray().flatten()
    possible_b_indices = np.where(row_a > 0)[0]

    paths = []

    for idx_b in possible_b_indices:
        if idx_b == idx_a:
            continue
        p_ab = row_a[idx_b]

        row_b = prob_matrix[idx_b, :].toarray().flatten()
        possible_c_indices = np.where(row_b > 0)[0]

        for idx_c in possible_c_indices:
            if idx_c == idx_a or idx_c == idx_b:
                continue
            p_bc = row_b[idx_c]
            total_prob = p_ab * p_bc

            paths.append({
                "path": f"{start_word} -> {idx_to_word[idx_b]} -> {idx_to_word[idx_c]}",
                "prob": total_prob,
                "target": idx_to_word[idx_c]
            })

    sorted_paths = sorted(paths, key=lambda x: x['prob'], reverse=True)
    return sorted_paths[:top_n]


def find_group_inference_paths(prob_matrix, vocab_dict, input_words, top_n=5):
    """
    多词联合 2 跳推理路径搜索。
    输入一组关键词，找出这组词共同指向的最强推理目标。

    核心设计：对同一个目标词，来自不同输入词的路径得分会累加，
    因此被多个输入词同时指向的目标词置信度更高。

    参数:
        prob_matrix: 概率/PPMI矩阵
        vocab_dict: dict - 词到索引的映射
        input_words: list[str] - 输入关键词列表
        top_n: int - 返回前 N 个目标词
    """
    idx_to_word = {v: k for k, v in vocab_dict.items()}
    input_indices = set()
    for w in input_words:
        if w in vocab_dict:
            input_indices.add(vocab_dict[w])

    if not input_indices:
        return "输入词均不在词库中"

    results = {}

    for idx_start in input_indices:
        start_word = idx_to_word[idx_start]

        row_start = prob_matrix[idx_start, :].toarray().flatten()
        mid_indices = np.where(row_start > 0.01)[0]

        for idx_mid in mid_indices:
            if idx_mid in input_indices:
                continue
            p1 = row_start[idx_mid]
            mid_word = idx_to_word[idx_mid]

            row_mid = prob_matrix[idx_mid, :].toarray().flatten()
            target_indices = np.where(row_mid > 0.01)[0]

            for idx_target in target_indices:
                if idx_target in input_indices:
                    continue

                p2 = row_mid[idx_target]
                path_score = p1 * p2

                if idx_target not in results:
                    results[idx_target] = {"total_score": 0, "explanations": []}

                results[idx_target]["total_score"] += path_score
                results[idx_target]["explanations"].append({
                    "from": start_word,
                    "bridge": mid_word,
                    "score": round(path_score, 6)
                })

    final_output = []
    for idx_target, data in results.items():
        sorted_expl = sorted(data["explanations"], key=lambda x: x["score"], reverse=True)
        final_output.append({
            "target": idx_to_word[idx_target],
            "confidence": round(data["total_score"], 6),
            "why": sorted_expl[:3]
        })

    final_output = sorted(final_output, key=lambda x: x["confidence"], reverse=True)
    return _merge_substring_targets(final_output[:top_n * 2], top_n)


def beam_search_inference(prob_matrix, vocab_dict, input_words,
                          max_depth=3, beam_width=20,
                          min_prob_threshold=0.01, top_n=10,
                          hop_decay=0.5, diversity_penalty=0.6,
                          hub_penalty_percentile=95):
    """
    N 跳自适应 Beam Search 推理（v3.1 - 多源必达 + 枢纽惩罚）。

    核心过滤逻辑（v3.1 改进）：
    当输入词 >= 2 时，目标词必须满足以下之一才保留：
      A) 路径源全覆盖：beam search 路径从每个输入词都能到达该目标
         (例: "市场" 从 "中国" 和 "出口" 都有路径到达 → 保留)
      B) 邻居全覆盖：目标词是所有输入词的直接邻居
         (即使路径只从部分输入词出发，但共现矩阵证明它与所有输入词相关)
    两项都不满足则过滤。这会杀死地震词：
      "震源" 只从 "中国" 有路径，且不是 "出口" 的直接邻居 → 过滤掉

    其他机制：
    - 跳数衰减（hop_decay）
    - 枢纽节点惩罚（高度数节点的路径降权）
    - 桥接多样性惩罚
    - 多源路径奖励

    参数:
        prob_matrix: 概率/PPMI矩阵
        vocab_dict: dict - 词到索引的映射
        input_words: list[str] - 输入关键词列表
        max_depth: int - 最大跳数
        beam_width: int - 每层保留的最大路径数
        min_prob_threshold: float - 路径累积概率低于此值则剪掉
        top_n: int - 最终返回的目标词数量
        hop_decay: float - 每跳的衰减因子（0~1），越小衰减越快
        diversity_penalty: float - 桥接节点单一时的惩罚系数（0~1）
        hub_penalty_percentile: int - 枢纽节点惩罚的度数百分位阈值
    """
    idx_to_word = {v: k for k, v in vocab_dict.items()}
    input_indices = set()
    for w in input_words:
        if w in vocab_dict:
            input_indices.add(vocab_dict[w])

    if not input_indices:
        return "输入词均不在词库中"

    n_inputs = len(input_indices)

    # --- 预计算1：每个输入词各自的直接邻居集合（用于双向相关性）---
    per_input_neighbors = {}
    for idx_s in input_indices:
        row_s = prob_matrix[idx_s, :].toarray().flatten()
        neighbors = set(np.where(row_s > 0)[0])
        per_input_neighbors[idx_s] = neighbors

    # --- 预计算2：节点度数 & 枢纽节点识别 ---
    # 计算每个节点的出度（非零连接数）
    node_degrees = np.diff(prob_matrix.indptr) if hasattr(prob_matrix, 'indptr') else None
    hub_threshold = None
    hub_nodes = set()
    if node_degrees is not None and len(node_degrees) > 0:
        hub_threshold = np.percentile(node_degrees[node_degrees > 0], hub_penalty_percentile)
        hub_nodes = set(np.where(node_degrees > hub_threshold)[0])

    hub_penalty_factor = 0.3  # 经过枢纽节点的边权惩罚系数

    # 存储每个目标词的汇总信息
    target_scores = {}

    # 对每个输入词分别做 Beam Search
    for idx_start in input_indices:
        start_word = idx_to_word[idx_start]

        # 当前层的活跃路径
        # 每条路径: (累积概率, 路径词列表, 当前末端节点索引, 第1跳桥接词)
        active_paths = [(1.0, [start_word], idx_start, None)]

        for depth in range(max_depth):
            next_paths = []
            # 本层的衰减因子
            layer_decay = hop_decay ** depth

            for cum_prob, path_words, current_idx, first_bridge in active_paths:
                # 获取当前节点的所有后继
                row = prob_matrix[current_idx, :].toarray().flatten()
                nonzero_indices = np.where(row > min_prob_threshold)[0]

                for idx_next in nonzero_indices:
                    if idx_next in input_indices:
                        continue

                    next_word = idx_to_word.get(idx_next)
                    if next_word is None or next_word in path_words:
                        continue

                    # 应用跳数衰减
                    edge_prob = row[idx_next] * layer_decay

                    # 枢纽节点惩罚：如果当前节点是枢纽词，降低通过它的边权
                    if current_idx in hub_nodes:
                        edge_prob *= hub_penalty_factor

                    new_cum_prob = cum_prob * edge_prob

                    if new_cum_prob < min_prob_threshold * 0.1:
                        continue

                    # 记录第1跳的桥接词
                    new_bridge = first_bridge if first_bridge is not None else next_word

                    new_path = path_words + [next_word]
                    next_paths.append((new_cum_prob, new_path, idx_next, new_bridge))

                    # 只在最终深度层计分
                    if depth == max_depth - 1:
                        if idx_next not in target_scores:
                            target_scores[idx_next] = {
                                "total_score": 0,
                                "paths": [],
                                "bridges": set(),
                                "source_words": set()
                            }

                        target_scores[idx_next]["total_score"] += new_cum_prob
                        target_scores[idx_next]["bridges"].add(new_bridge)
                        target_scores[idx_next]["source_words"].add(start_word)
                        target_scores[idx_next]["paths"].append({
                            "from": start_word,
                            "chain": " -> ".join(new_path),
                            "depth": depth + 1,
                            "score": round(new_cum_prob, 6),
                            "bridge": new_bridge
                        })

            if not next_paths:
                break
            next_paths.sort(key=lambda x: x[0], reverse=True)
            active_paths = next_paths[:beam_width]

    # --- 后处理 ---
    final_output = []
    for idx_target, data in target_scores.items():
        target_word = idx_to_word[idx_target]

        # === 核心过滤：多源可达性 + 邻居覆盖度联合判定 ===

        # 1. 计算目标词是多少个输入词的直接邻居（邻居覆盖度）
        n_connected_inputs = 0
        for idx_s in input_indices:
            if idx_target in per_input_neighbors[idx_s]:
                n_connected_inputs += 1
        coverage = n_connected_inputs / n_inputs  # 0~1

        # 2. 计算路径来自多少个不同的输入词（路径源覆盖度）
        n_sources = len(data["source_words"])
        source_coverage = n_sources / n_inputs  # 0~1

        # 3. 联合过滤规则（多输入词场景）
        if n_inputs >= 2:
            # 硬过滤：目标词必须同时满足以下之一才保留：
            #   A) 路径来自所有输入词（source_coverage == 1.0），说明每个输入词
            #      都能通过多跳到达它 —— 这是最强的相关性信号
            #   B) 路径来自部分输入词，但目标词是所有输入词的直接邻居
            #      (coverage == 1.0)，说明它与所有输入词都有直接共现关系
            if source_coverage < 1.0 and coverage < 1.0:
                # 两项都不满足 → 该目标只与部分输入词相关，过滤
                continue

        # 4. 单输入词场景：必须是该输入词的直接邻居
        if n_inputs == 1 and n_connected_inputs == 0:
            continue

        score = data["total_score"]

        # === 覆盖度加权 ===
        # 全覆盖(coverage=1.0)的目标得满分；
        # 只与部分输入词有直接关系的按比例降权
        coverage_weight = coverage ** 0.5  # 平方根使得部分覆盖不至于降太多
        score *= coverage_weight

        # === 多源路径奖励 ===
        # 所有输入词都能到达的目标获得奖励
        if n_sources >= n_inputs and n_inputs >= 2:
            score *= 1.5  # 全源可达 +50%
        elif n_sources > 1:
            score *= (1.0 + 0.2 * (n_sources - 1))  # 部分多源 +20% per source

        # === 桥接多样性惩罚 ===
        n_bridges = len(data["bridges"])
        if n_bridges <= 1:
            score *= diversity_penalty

        sorted_paths = sorted(data["paths"], key=lambda x: x["score"], reverse=True)
        clean_paths = [{k: v for k, v in p.items() if k != "bridge"} for p in sorted_paths[:3]]

        final_output.append({
            "target": target_word,
            "confidence": round(score, 6),
            "why": clean_paths
        })

    final_output.sort(key=lambda x: x["confidence"], reverse=True)
    return _merge_substring_targets(final_output[:top_n * 2], top_n)
