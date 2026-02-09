# -*- coding: utf-8 -*-
"""
增量共现矩阵管理器
- 全局矩阵增量累加，替代 deepcopy + list 存储
- 内存从 O(N * nnz) 降到 O(nnz)
"""
from scipy.sparse import dok_matrix, csr_matrix
import numpy as np


class IncrementalCooccurrence:
    """
    增量式词共现矩阵管理器。

    核心改进：每条文档直接在全局矩阵上 += 1，
    不再为每条文档 deepcopy 一份完整矩阵。
    """

    def __init__(self, max_dim=100000):
        self.max_dim = max_dim
        self.matrix = dok_matrix((max_dim, max_dim), dtype=np.float64)
        self.vocab_dict = {}      # word -> idx
        self.idx_to_word = {}     # idx -> word
        self.vocab_count = 0
        self.total_docs = 0

    def _get_or_create_idx(self, word):
        """获取词的索引，如果不存在则自动创建"""
        if word not in self.vocab_dict:
            idx = self.vocab_count
            self.vocab_dict[word] = idx
            self.idx_to_word[idx] = word
            self.vocab_count += 1
            return idx
        return self.vocab_dict[word]

    def add_document(self, words):
        """
        处理一条文档的词列表，直接增量累加到全局共现矩阵。

        参数:
            words: list[str] - 经过分词和去重后的词列表
        """
        # 获取所有词的索引
        indices = [self._get_or_create_idx(w) for w in words]

        # 直接在全局矩阵上累加共现关系
        for i in indices:
            for j in indices:
                self.matrix[i, j] += 1

        self.total_docs += 1

    def add_keywords(self, keywords):
        """
        从记忆条目的关键词列表学习共现关系。

        与 add_document 的区别：
        - keywords 已经是干净的关键词，无需分词
        - 语义密度更高（每个词都是重要概念）
        - 不计入 total_docs（避免影响 PPMI 的文档频率计算基准）

        参数:
            keywords: list[str] - 关键词列表
        """
        if not keywords or len(keywords) < 2:
            return

        indices = [self._get_or_create_idx(w) for w in keywords]

        for i in indices:
            for j in indices:
                self.matrix[i, j] += 1

    def get_csr_matrix(self):
        """将当前矩阵转换为 CSR 格式（高效计算用）"""
        return self.matrix.tocsr()

    def prune(self, min_cooccurrence=10):
        """
        剪枝：去除共现次数低于阈值的元素。

        参数:
            min_cooccurrence: int - 最小共现次数阈值
        返回:
            csr_matrix - 剪枝后的 CSR 矩阵
        """
        m_csr = self.matrix.tocsr().copy()
        m_csr.data[m_csr.data < min_cooccurrence] = 0
        m_csr.eliminate_zeros()
        return m_csr

    def get_stats(self):
        """返回当前矩阵的统计信息"""
        csr = self.get_csr_matrix()
        return {
            "total_docs": self.total_docs,
            "vocab_size": self.vocab_count,
            "nonzero_pairs": csr.nnz,
            "matrix_density": csr.nnz / (self.vocab_count ** 2) if self.vocab_count > 0 else 0,
        }

    def remove_words(self, words_to_remove):
        """
        从共现矩阵中移除指定词汇，缩减矩阵维度。

        用于 consolidate() 时的词汇剪枝。被移除的词：
        - 从 vocab_dict / idx_to_word 中删除
        - 对应的矩阵行列被清零
        - vocab_count 相应减小
        - 矩阵索引被重新压缩

        注意: 此操作不影响 SQLite 中的记忆条目。被剪枝的词
        仍可通过精确关键词匹配（fast query）找到对应条目。

        参数:
            words_to_remove: list[str] - 要移除的词汇列表

        返回:
            int - 实际移除的词汇数量
        """
        if not words_to_remove:
            return 0

        # 找出要移除的索引
        indices_to_remove = set()
        for w in words_to_remove:
            if w in self.vocab_dict:
                indices_to_remove.add(self.vocab_dict[w])

        if not indices_to_remove:
            return 0

        # 构建保留的索引列表（保持原始顺序）
        indices_to_keep = [i for i in range(self.vocab_count)
                           if i not in indices_to_remove]

        if not indices_to_keep:
            # 全部移除 = 重置
            removed = self.vocab_count
            self.matrix = dok_matrix(
                (self.max_dim, self.max_dim), dtype=np.float64)
            self.vocab_dict = {}
            self.idx_to_word = {}
            self.vocab_count = 0
            return removed

        # 提取保留部分的子矩阵
        csr = self.matrix.tocsr()
        # 取出保留行列的交叉子矩阵
        sub = csr[indices_to_keep, :][:, indices_to_keep]

        # 重建 vocab 映射
        new_vocab_dict = {}
        new_idx_to_word = {}
        for new_idx, old_idx in enumerate(indices_to_keep):
            word = self.idx_to_word[old_idx]
            new_vocab_dict[word] = new_idx
            new_idx_to_word[new_idx] = word

        # 重建 DOK 矩阵
        new_count = len(indices_to_keep)
        new_dok = dok_matrix((self.max_dim, self.max_dim), dtype=np.float64)

        # 将子矩阵数据填入新 DOK
        sub_coo = sub.tocoo()
        for r, c, v in zip(sub_coo.row, sub_coo.col, sub_coo.data):
            new_dok[r, c] = v

        removed = self.vocab_count - new_count

        # 更新状态
        self.matrix = new_dok
        self.vocab_dict = new_vocab_dict
        self.idx_to_word = new_idx_to_word
        self.vocab_count = new_count

        return removed
