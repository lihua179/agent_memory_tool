# -*- coding: utf-8 -*-
"""
@author: Zed
@file: d.py
@time: 2026/2/9 2:32
@describe:自定义描述
"""
import time

from quant_tool.tool import get_file
import pandas as pd
import jieba
import jieba.analyse as analyse
from scipy.sparse import dok_matrix
from copy import deepcopy
from scipy.sparse import diags
import numpy as np
from scipy.sparse import csr_matrix
from multi_words_search_roads import find_top_inference_paths




# 初始化一个逻辑上很大但实际不占空间的稀疏矩阵
# 假设最大支持 100万个词
max_dim = 100000
sparse_memory = dok_matrix((max_dim, max_dim), dtype=int)
# zero_list=np.zeros(max_dim, dtype=int)
# print(zero_list)
# time.sleep(1000)
# 更新时，只针对有值的坐标进行操作
# sparse_memory[i, j] += 1
base_path = fr"D:\sina_news\sina_news_data\tag_0"
file_list = get_file(base_path)
file_list = file_list[:2]
# print(file_list)

# elf.vocab = vocab_list
#         self.word2idx = {word: i for i, word in enumerate(vocab_list)}
#         self.dim = len(vocab_list)
# {word: i for i, word in enumerate(vocab_list)
vocab_dict = {}
vocab_count = 0
sparse_memory_list = []
total_news = 0
from tqdm import tqdm

for file in file_list:
    df = pd.read_csv(base_path + '\\' + file)
    print(df)
    for i in tqdm(range(len(df))):
        total_news += 1
        d = df.iloc[i].values[3]
        # print(d)
        # 1. 原始文本数据
        # texts = [d]
        text = d
        # print("===== 方案一：基础分词（精确模式） =====")
        # for i, text in enumerate(texts):
        # jieba.cut 生成一个生成器，用 lcut 直接转为列表
        words = jieba.cut(text, cut_all=True)
        # 过滤掉标点符号和单字（单字通常没有太强的属性意义）
        clean_words = [w for w in words if len(w) > 1]
        clean_words = list(set(clean_words))
        # print(f"任务 {i + 1} 词组: {clean_words}")
        sparse_memory_copy = deepcopy(sparse_memory)
        index_list = []
        for word in clean_words:
            if word not in vocab_dict:
                vocab_dict[word] = vocab_count
                vocab_count += 1
            index_list.append(vocab_dict[word])
            # zero_list_copy[vocab_dict[word]]=1
        for i in index_list:
            for j in index_list:
                sparse_memory_copy[i, j] = 1
        #
        # print(sparse_memory_copy)
        if sparse_memory_list:
            # 第n词，向量库存储够了，我们开始推理吧！我们开始统计这次的累积概率。我们首先进行累加操作
            if total_news >= 2000:
                now_csr_sparse_memory = sparse_memory_copy.tocsr()
                cumulate_sparse_memory = now_csr_sparse_memory
                for sparse_memory in sparse_memory_list:
                    # cumulate_sparse_memory = cumulate_sparse_memory + sparse_memory.tocsr() - total_news * 0.000001  # 每条新闻衰减系数（每十万条新闻-1）
                    cumulate_sparse_memory = cumulate_sparse_memory + sparse_memory.tocsr()

                # 当前所有累积新闻的稀疏矩阵累加
                # 算到这一步，已经统计完每两个词之间同时在新闻中出现的次数的累积矩阵了
                cumulate_sparse_memory = cumulate_sparse_memory.todok()

                # 下一步，我们计算每个俩俩组成的元素，它们相对各自元素（两个）同时出现的概率矩阵
                # # 1. 提取对角线作为分母 (A出现的总次数)
                diag = cumulate_sparse_memory.diagonal()
                #
                # # 2. 为了防止除以 0，把 0 换成 1
                diag[diag == 0] = 1
                #
                # 3. 计算概率矩阵 P
                # 原理：矩阵的每一行 i，除以该行的对角线元素 M[i,i]
                # 得到的结果 P[i,j] 就是：看到词 i 后，有多大概念能想到词 j

                inv_diag_matrix = diags(1.0 / diag)
                prob_matrix = inv_diag_matrix @ sparse_memory.tocsr()
                #
                # 现在，prob_matrix[idx_A, idx_B] 就是从 A 推导 B 的概率
                print(prob_matrix.shape[0])
                start_word = '美国'
                res = find_top_inference_paths(prob_matrix, vocab_dict, start_word, top_n=2)
                print(res)
                time.sleep(1000)

        # 每次条新闻产生的关键词稀疏矩阵
        sparse_memory_list.append(sparse_memory_copy)
        # m_current = np.outer(zero_list_copy, zero_list_copy)
        # print(m_current)
        #
        # 提取这次所有词的索引，做成一维向量

        # 然后展开为二维矩阵，然后把这个二维矩阵与旧的累积二维矩阵进行累加（考虑时间衰减，要把每单次新闻计算的二维矩阵进行存储，
        # 然后每次重新计算多个加时间衰减权重的二维矩阵的累积值），就是当前最新的累积的相关矩阵

        # sparse_memory[i, vocab_dict[word]] = 1

# print(sparse_memory)
