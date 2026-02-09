# -*- coding: utf-8 -*-
"""
概率矩阵计算模块
- PPMI（正值点互信息）：过滤高频虚词噪声，比条件概率更准确
- 条件概率矩阵：保留原始方案兼容
"""
import numpy as np
from scipy.sparse import diags, csr_matrix


def compute_conditional_prob_matrix(cooccurrence_csr):
    """
    计算条件概率矩阵（兼容原始方案）。

    P[i,j] = M[i,j] / M[i,i]
    即：看到词 i 后，有多大概率能想到词 j

    参数:
        cooccurrence_csr: csr_matrix - 共现矩阵（已剪枝）
    返回:
        csr_matrix - 条件概率矩阵
    """
    diag = cooccurrence_csr.diagonal().copy().astype(np.float64)
    diag[diag == 0] = 1.0
    inv_diag = diags(1.0 / diag)
    prob_matrix = inv_diag @ cooccurrence_csr
    return prob_matrix


def compute_ppmi_matrix(cooccurrence_csr, total_docs):
    """
    计算 PPMI（正值点互信息）矩阵。

    PMI(A,B) = log2(P(A,B) / (P(A) * P(B)))
             = log2((M[A,B] * total_docs) / (M[A,A] * M[B,B]))

    PPMI = max(0, PMI)

    优势：
    - 有效过滤掉"记者""今日"等高频无意义词的虚假高共现
    - 突出真正有语义关联的词对
    - 在 NLP 领域是经过广泛验证的经典方法

    参数:
        cooccurrence_csr: csr_matrix - 共现矩阵（已剪枝）
        total_docs: int - 总文档数
    返回:
        csr_matrix - PPMI 矩阵
    """
    if total_docs == 0:
        return cooccurrence_csr.copy()

    # 提取对角线：每个词出现的文档数
    diag = cooccurrence_csr.diagonal().copy().astype(np.float64)

    # 将 COO 格式遍历每个非零元素，计算 PMI
    coo = cooccurrence_csr.tocoo()

    new_data = []
    new_row = []
    new_col = []

    for r, c, v in zip(coo.row, coo.col, coo.data):
        # 跳过对角线（词与自身的共现没有意义）
        if r == c:
            continue

        p_a = diag[r]  # 词A出现的文档数
        p_b = diag[c]  # 词B出现的文档数

        if p_a == 0 or p_b == 0:
            continue

        # PMI = log2((M[A,B] * total_docs) / (M[A,A] * M[B,B]))
        pmi = np.log2((v * total_docs) / (p_a * p_b))

        # PPMI: 只保留正值
        if pmi > 0:
            new_data.append(pmi)
            new_row.append(r)
            new_col.append(c)

    ppmi_matrix = csr_matrix(
        (new_data, (new_row, new_col)),
        shape=cooccurrence_csr.shape
    )
    return ppmi_matrix


def print_top_associations(prob_matrix, idx_to_word, min_score=0.2, max_score=1.0, top_n=50):
    """
    打印最强关联词对。

    参数:
        prob_matrix: 概率/PPMI矩阵
        idx_to_word: dict - 索引到词的映射
        min_score: float - 最小分数阈值
        max_score: float - 最大分数阈值（排除对角线自关联）
        top_n: int - 最多打印多少条
    """
    coo = prob_matrix.tocoo()

    pairs = []
    for r, c, v in zip(coo.row, coo.col, coo.data):
        if r == c:
            continue
        if min_score <= v < max_score:
            if r in idx_to_word and c in idx_to_word:
                pairs.append((idx_to_word[r], idx_to_word[c], v))

    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\n===== Top {min(top_n, len(pairs))} 强关联词对 =====")
    for word_a, word_b, score in pairs[:top_n]:
        print(f"  {word_a} -> {word_b}: {score:.4f}")
    print(f"  (共 {len(pairs)} 条超过阈值 {min_score})")
