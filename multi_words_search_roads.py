# -*- coding: utf-8 -*-
"""
@author: Zed
@file: multi_words_search_roads.py
@time: 2026/2/9 4:00
@describe:词组推理路径搜索
"""
import numpy as np

def find_top_inference_paths(prob_matrix, vocab_dict, start_word, top_n=5):
    """
    单词组推理路径搜索，寻找从起点出发的最强 2步推理路径 (A -> B -> C)
    """
    if start_word not in vocab_dict:
        return "词不在词库中"

    idx_a = vocab_dict[start_word]
    idx_to_word = {v: k for k, v in vocab_dict.items()}

    # 1. 找到所有 A 能够直接推导出的中间词 B
    # row_a 存储了 P[idx_a, :]
    row_a = prob_matrix[idx_a, :].toarray().flatten()
    possible_b_indices = np.where(row_a > 0)[0]

    paths = []

    # 2. 遍历每个中间词 B，寻找 C
    for idx_b in possible_b_indices:
        p_ab = row_a[idx_b]

        # 找到 B 能够推导出的词 C
        row_b = prob_matrix[idx_b, :].toarray().flatten()
        possible_c_indices = np.where(row_b > 0)[0]

        for idx_c in possible_c_indices:
            # 排除 A->B->A 这种循环
            if idx_c == idx_a: continue

            p_bc = row_b[idx_c]

            # 计算路径总概率
            total_prob = p_ab * p_bc

            paths.append({
                "path": f"{start_word} -> {idx_to_word[idx_b]} -> {idx_to_word[idx_c]}",
                "prob": total_prob,
                "target": idx_to_word[idx_c]
            })

    # 3. 按概率排序
    sorted_paths = sorted(paths, key=lambda x: x['prob'], reverse=True)

    return sorted_paths[:top_n]
def find_group_inference_paths(prob_matrix, vocab_dict, input_words, top_n=5):
    """
    多词组推理路径搜索，输入一组关键词，找出这组词共同指向的最强推理路径
    """
    idx_to_word = {v: k for k, v in vocab_dict.items()}
    input_indices = [vocab_dict[w] for w in input_words if w in vocab_dict]

    if not input_indices:
        return "输入词均不在词库中"

    # 存储所有发现的路径
    # 格式: {目标词idx: {score: 总分, paths: [路径详情列表]}}
    results = {}

    for idx_start in input_indices:
        start_word = idx_to_word[idx_start]

        # 1. 找到直接关联词 (第一步)
        row_start = prob_matrix[idx_start, :].toarray().flatten()
        mid_indices = np.where(row_start > 0.01)[0]  # 设定阈值过滤微小噪音

        for idx_mid in mid_indices:
            p1 = row_start[idx_mid]
            mid_word = idx_to_word[idx_mid]

            # 2. 找到推理关联词 (第二步)
            row_mid = prob_matrix[idx_mid, :].toarray().flatten()
            target_indices = np.where(row_mid > 0.01)[0]

            for idx_target in target_indices:
                # 排除输入词本身
                if idx_target in input_indices: continue

                p2 = row_mid[idx_target]
                target_word = idx_to_word[idx_target]

                # 计算这条路径的得分
                path_score = p1 * p2

                if idx_target not in results:
                    results[idx_target] = {"total_score": 0, "explanations": []}

                # 累加得分：如果多个输入词都指向它，分值会很高
                results[idx_target]["total_score"] += path_score
                results[idx_target]["explanations"].append({
                    "from": start_word,
                    "bridge": mid_word,
                    "score": path_score
                })

    # 3. 整理并排序
    final_output = []
    for idx_target, data in results.items():
        # 对解释路径按得分排序，只取对该目标贡献最大的前2条路径
        sorted_expl = sorted(data["explanations"], key=lambda x: x["score"], reverse=True)

        final_output.append({
            "target": idx_to_word[idx_target],
            "confidence": data["total_score"],
            "why": sorted_expl[:2]  # 展示主要的推理逻辑
        })

    # 按总置信度排序
    final_output = sorted(final_output, key=lambda x: x["confidence"], reverse=True)

    return final_output[:top_n]

# ==========================================
# 演示效果
# ==========================================
# 假设输入: ["特朗普", "马斯克"]
# 输出可能如下:
# 1. 目标: "DOGE" (置信度: 0.85)
#    - 路径 A: 特朗普 -> 政府效率部 -> DOGE (0.45)
#    - 路径 B: 马斯克 -> 加密货币 -> DOGE (0.40)
#
# 2. 目标: "关税" (置信度: 0.65)
#    - 路径 A: 特朗普 -> 贸易政策 -> 关税 (0.50)
#    - 路径 B: 马斯克 -> 特斯拉成本 -> 关税 (0.15)