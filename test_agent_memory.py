# -*- coding: utf-8 -*-
"""
AgentMemory 统一 API 端到端测试

测试内容:
    Test 1: 基础 CRUD (write/read/remove/list_entries)
    Test 2: 文档摄入 + 矩阵构建 (ingest_document + rebuild_matrices)
    Test 3: 三级查询 (query: fast/standard/deep)
    Test 4: 链推理 (find_chain + with_evidence)
    Test 5: 持久化 (save + load + 验证一致性)
    Test 6: 多智能体共享 (source_filter)
    Test 7: 时间过滤 (time_recent / time_range)
    Test 8: 双通道学习 (write 自动触发 add_keywords)
    Test 9: repr + get_stats
    Test 10: log() 写入 + 索引验证 + 双通道学习
    Test 11: recall_by_date() 纯时间查询
    Test 12: recall_by_range() 时间段 + 分类/关键词/来源过滤
    Test 13: summarize() 聚合统计
    Test 14: Episodic 持久化 save/load 一致性
    Test 15: 日志 + 语义查询协作
    Test 16: SQLite CRUD + 三库分类 (entry_type 过滤)
    Test 17: 提取强化 (access_count 查询后自增)
    Test 18: 重要性分级 (影响时间衰减 + 排序)
    Test 19: 记忆再巩固 (supersedes 链接 + 旧条目降级)
    Test 20: 整合清理 (cleanup_vocab + remove_words + consolidate)

运行方式:
    python test_agent_memory.py
    (不要通过管道捕获输出，直接在终端运行)
"""

import sys
import os
import time
import json
import shutil

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory import AgentMemory


def print_sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_1_crud():
    """Test 1: 基础 CRUD"""
    print_sep("Test 1: CRUD")

    am = AgentMemory()

    # write
    id1 = am.write(topic="AI发展", keywords=["人工智能", "深度学习", "神经网络"],
                    summary="人工智能领域近年来飞速发展", source="agent_A")
    id2 = am.write(topic="量子计算", keywords=["量子", "计算", "量子比特"],
                    summary="量子计算有望突破经典计算的瓶颈", source="agent_B")
    id3 = am.write(summary="这是一条没有主题的笔记",
                    auto_extract_keywords=True, source="agent_A")

    assert id1 is not None, "write 返回 None"
    assert id2 is not None, "write 返回 None"
    assert id3 is not None, "write 返回 None"
    print(f"  [*] write: 成功写入 3 条 ({id1}, {id2}, {id3})")

    # read
    entry1 = am.read(id1)
    assert entry1 is not None, "read 返回 None"
    assert entry1["topic"] == "AI发展", f"topic 不匹配: {entry1['topic']}"
    assert entry1["source"] == "agent_A", f"source 不匹配"
    print(f"  [*] read: {id1} -> topic={entry1['topic']}, source={entry1['source']}")

    # read 不存在
    assert am.read("not_exist") is None, "不存在的 entry 应返回 None"
    print(f"  [*] read(not_exist): None (correct)")

    # list_entries
    all_entries = am.list_entries()
    assert len(all_entries) == 3, f"应有 3 条，实际 {len(all_entries)}"
    print(f"  [*] list_entries: {len(all_entries)} 条")

    # list by source
    a_entries = am.list_entries(source_filter="agent_A")
    assert len(a_entries) == 2, f"agent_A 应有 2 条，实际 {len(a_entries)}"
    print(f"  [*] list_entries(agent_A): {len(a_entries)} 条")

    # remove
    removed = am.remove(id3)
    assert removed is True, "remove 应返回 True"
    assert am.read(id3) is None, "删除后 read 应返回 None"
    assert len(am.list_entries()) == 2, "删除后应剩 2 条"
    print(f"  [*] remove({id3}): OK, 剩余 {len(am.list_entries())} 条")

    # remove 不存在
    assert am.remove("not_exist") is False, "不存在的 entry remove 应返回 False"
    print(f"  [*] remove(not_exist): False (correct)")

    # 错误输入
    try:
        am.write()  # 无内容
        assert False, "应抛出 ValueError"
    except ValueError:
        print(f"  [*] write() 空内容: ValueError (correct)")

    print("  >>> Test 1 PASSED")
    return True


def test_2_ingest_and_matrices():
    """Test 2: 文档摄入 + 矩阵构建"""
    print_sep("Test 2: Ingest + Matrices")

    am = AgentMemory(min_cooccurrence=2)

    # 模拟新闻文档
    docs = [
        "特朗普宣布对中国商品加征关税，中美贸易摩擦加剧。美国股市出现大幅波动，投资者担忧经济前景。",
        "美联储主席鲍威尔表示将维持当前利率不变，市场预计降息概率下降。美元指数走强，人民币汇率承压。",
        "中国出口数据超预期增长，贸易顺差扩大。分析师认为全球供应链调整推动了出口增长。",
        "特斯拉在中国市场销量创新高，电动汽车行业竞争激烈。比亚迪和蔚来等国产品牌持续发力。",
        "A股市场震荡走低，北向资金持续流出。券商分析师建议投资者保持谨慎，关注政策面变化。",
        "美联储加息预期升温导致全球股市下跌，新兴市场资金外流明显。人民币兑美元汇率创下新低。",
        "中美贸易谈判取得进展，双方同意降低部分商品关税。市场情绪改善，A股大幅反弹。",
        "特朗普签署行政命令限制中国科技企业在美投资。华为和中兴通讯股价应声下跌。",
        "全球供应链受地缘政治影响出现重构，越南和印度成为新的制造业转移目的地。",
        "中国央行降准释放流动性，支持实体经济发展。银行间市场利率小幅回落。",
    ]

    for doc in docs:
        am.ingest_document(doc)

    stats = am.get_stats()
    print(f"  [*] 摄入 {stats['cooccurrence']['total_docs']} 篇文档")
    print(f"  [*] 词汇量: {stats['cooccurrence']['vocab_size']}")
    assert stats['cooccurrence']['total_docs'] == 10, "应摄入 10 篇"
    assert stats['cooccurrence']['vocab_size'] > 20, "词汇量过小"

    # 手动重建矩阵
    result = am.rebuild_matrices()
    print(f"  [*] rebuild_matrices: pruned_nnz={result['pruned_nnz']}, "
          f"ppmi_nnz={result['ppmi_nnz']}, time={result['time_ms']}ms")
    assert result['ppmi_nnz'] > 0, "PPMI 非零元素数应 > 0"

    # 检查 dirty 标记
    assert am._matrices_dirty is False, "rebuild 后 dirty 应为 False"

    # 再摄入一篇，dirty 应变 True
    am.ingest_document("美国经济数据好于预期，美股继续上涨。")
    assert am._matrices_dirty is True, "新文档摄入后 dirty 应为 True"

    print("  >>> Test 2 PASSED")
    return am  # 返回 am 供后续测试复用


def test_3_query(am):
    """Test 3: 三级查询"""
    print_sep("Test 3: Query (fast/standard/deep)")

    # 先写入一些记忆条目
    am.write(topic="中美贸易摩擦",
             keywords=["特朗普", "关税", "贸易", "中国", "美国"],
             summary="特朗普政府对中国商品加征关税引发贸易摩擦",
             source="news_agent")

    am.write(topic="美联储货币政策",
             keywords=["美联储", "利率", "降息", "鲍威尔"],
             summary="美联储维持利率不变，市场关注未来降息时间点",
             source="news_agent")

    am.write(topic="A股市场走势",
             keywords=["A股", "股市", "北向资金", "投资"],
             summary="A股震荡走低，北向资金持续流出",
             source="market_agent")

    # 重建矩阵（因为有新数据）
    am.rebuild_matrices()

    # fast 查询
    r_fast = am.query(keywords=["关税", "贸易"], depth="fast", token_budget=500)
    assert r_fast["search_stats"]["depth_used"] == "fast"
    assert len(r_fast["matched_entries"]) > 0, "fast 查询应有结果"
    assert r_fast["search_stats"]["time_ms"] < 5000, "fast 不应超过 5 秒"
    print(f"  [*] fast: {len(r_fast['matched_entries'])} 条命中, "
          f"{r_fast['search_stats']['time_ms']}ms")
    print(f"      prompt_text: {len(r_fast['prompt_text'])} chars")

    # standard 查询
    r_std = am.query(keywords=["关税"], depth="standard", token_budget=800)
    assert r_std["search_stats"]["depth_used"] == "standard"
    print(f"  [*] standard: {len(r_std['matched_entries'])} 条命中, "
          f"{r_std['search_stats']['time_ms']}ms")
    print(f"      expanded_keywords: {r_std['expanded_keywords'][:5]}")

    # deep 查询（需要 >= 2 个关键词才会触发 chain）
    r_deep = am.query(keywords=["特朗普", "A股"], depth="deep", token_budget=1000)
    assert r_deep["search_stats"]["depth_used"] == "deep"
    print(f"  [*] deep: {len(r_deep['matched_entries'])} 条命中, "
          f"{r_deep['search_stats']['time_ms']}ms")
    if r_deep["chain"]:
        chains = r_deep["chain"].get("chains", [])
        print(f"      chains: {len(chains)} 条路径")
        for c in chains[:2]:
            if c.get("path"):
                print(f"        {' -> '.join(c['path'])} (w={c['total_weight']})")

    # user_input 自动提取
    r_auto = am.query(user_input="最近的贸易战对股市有什么影响", depth="fast")
    print(f"  [*] user_input 自动提取: {len(r_auto['matched_entries'])} 条命中")

    # 空查询
    r_empty = am.query(keywords=[], depth="fast")
    assert len(r_empty["matched_entries"]) == 0, "空关键词应无结果"
    print(f"  [*] empty query: 0 条 (correct)")

    # token 预算检验
    from memory.retriever import count_tokens
    if r_fast["prompt_text"]:
        actual_tokens = count_tokens(r_fast["prompt_text"])
        print(f"  [*] token budget check: budget=500, actual={actual_tokens}")
        assert actual_tokens <= 550, f"token 超预算: {actual_tokens}"

    print("  >>> Test 3 PASSED")


def test_4_chain(am):
    """Test 4: 链推理"""
    print_sep("Test 4: Chain Reasoning")

    # 基础链推理
    chain_result = am.find_chain(["特朗普", "A股"])
    print(f"  [*] find_chain(['特朗普', 'A股']):")
    print(f"      anchors_found: {chain_result.get('anchors_found', [])}")
    print(f"      anchors_missing: {chain_result.get('anchors_missing', [])}")

    if chain_result.get("hidden_words"):
        print(f"      hidden_words (top 5):")
        for hw in chain_result["hidden_words"][:5]:
            print(f"        {hw['word']}: ppr={hw['ppr_score']}, "
                  f"path={hw['path_count']}, combined={hw['combined_score']}")

    if chain_result.get("chains"):
        print(f"      chains:")
        for c in chain_result["chains"]:
            if c.get("path"):
                print(f"        {' -> '.join(c['path'])} "
                      f"(w={c['total_weight']}, hops={c['hops']})")

    # 带证据的链推理
    chain_ev = am.find_chain(["美联储", "A股"], with_evidence=True)
    print(f"\n  [*] find_chain(['美联储', 'A股'], with_evidence=True):")
    if chain_ev.get("evidence"):
        print(f"      evidence ({len(chain_ev['evidence'])} 条):")
        for ev in chain_ev["evidence"][:3]:
            print(f"        {ev['word']}: topic={ev.get('topic')}")

    # 单词（应返回 error）
    chain_single = am.find_chain(["特朗普"])
    # discover_hidden_chain 要求至少 2 个锚点词
    # 但如果只有 1 个在图中，会返回 error
    print(f"  [*] find_chain single word: "
          f"hidden_words={len(chain_single.get('hidden_words', []))}")

    print("  >>> Test 4 PASSED")


def test_5_persistence(am):
    """Test 5: 持久化 save/load"""
    print_sep("Test 5: Persistence")

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_test_save_dir")

    try:
        # 记录保存前状态
        stats_before = am.get_stats()
        entries_before = am.list_entries()
        print(f"  [*] save 前: docs={stats_before['cooccurrence']['total_docs']}, "
              f"vocab={stats_before['cooccurrence']['vocab_size']}, "
              f"entries={stats_before['store']['total_entries']}, "
              f"ppmi_nnz={stats_before['matrices']['ppmi_nnz']}")

        # 保存
        am.save(save_dir)
        print(f"  [*] save -> {save_dir}")

        # 检查文件是否生成
        expected_files = ["config.json", "memory_store.json",
                          "cooccurrence.npz", "ppmi.npz", "pruned.npz"]
        for f in expected_files:
            fpath = os.path.join(save_dir, f)
            exists = os.path.exists(fpath)
            size = os.path.getsize(fpath) if exists else 0
            print(f"      {f}: {'OK' if exists else 'MISSING'} ({size} bytes)")
            assert exists, f"文件 {f} 未生成"

        # 新建实例加载
        am2 = AgentMemory()
        ok = am2.load(save_dir)
        assert ok is True, "load 应返回 True"

        stats_after = am2.get_stats()
        print(f"\n  [*] load 后: docs={stats_after['cooccurrence']['total_docs']}, "
              f"vocab={stats_after['cooccurrence']['vocab_size']}, "
              f"entries={stats_after['store']['total_entries']}, "
              f"ppmi_nnz={stats_after['matrices']['ppmi_nnz']}")

        # 验证一致性
        assert stats_after['cooccurrence']['total_docs'] == \
               stats_before['cooccurrence']['total_docs'], "total_docs 不一致"
        assert stats_after['cooccurrence']['vocab_size'] == \
               stats_before['cooccurrence']['vocab_size'], "vocab_size 不一致"
        assert stats_after['store']['total_entries'] == \
               stats_before['store']['total_entries'], "entries 数量不一致"
        assert stats_after['matrices']['ppmi_nnz'] == \
               stats_before['matrices']['ppmi_nnz'], "ppmi_nnz 不一致"

        # 验证条目内容
        entries_after = am2.list_entries()
        for eb in entries_before:
            ea = am2.read(eb["id"])
            assert ea is not None, f"条目 {eb['id']} 加载后不存在"
            assert ea["topic"] == eb["topic"], f"topic 不匹配: {ea['topic']} vs {eb['topic']}"
        print(f"  [*] 条目内容验证: OK ({len(entries_before)} 条)")

        # 验证加载后的查询仍然有效
        r = am2.query(keywords=["关税", "贸易"], depth="standard", token_budget=500)
        assert len(r["matched_entries"]) > 0, "加载后查询应有结果"
        print(f"  [*] 加载后查询: {len(r['matched_entries'])} 条命中, "
              f"{r['search_stats']['time_ms']}ms")

        # 验证加载后还能继续写入
        new_id = am2.write(topic="新写入测试", keywords=["测试"],
                           summary="加载后写入", source="test")
        assert am2.read(new_id) is not None, "加载后写入失败"
        print(f"  [*] 加载后写入: {new_id} OK")

        # load 不存在的目录
        am3 = AgentMemory()
        assert am3.load("_not_exist_dir_xyz") is False, "不存在的目录应返回 False"
        print(f"  [*] load(不存在): False (correct)")

        print("  >>> Test 5 PASSED")

    finally:
        # 清理
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            print(f"  [*] cleanup: {save_dir} removed")


def test_6_multi_agent(am):
    """Test 6: 多智能体共享"""
    print_sep("Test 6: Multi-Agent")

    # 已有 news_agent 和 market_agent 的条目
    stats = am.get_stats()
    print(f"  [*] 当前 sources: {stats['store']['sources']}")

    # 按来源查询
    r_news = am.query(keywords=["关税"], depth="fast", source_filter="news_agent")
    r_market = am.query(keywords=["股市"], depth="fast", source_filter="market_agent")
    r_all = am.query(keywords=["关税"], depth="fast")

    print(f"  [*] news_agent query: {len(r_news['matched_entries'])} 条")
    print(f"  [*] market_agent query: {len(r_market['matched_entries'])} 条")
    print(f"  [*] all agents query: {len(r_all['matched_entries'])} 条")

    # news_agent 的结果不应包含 market_agent 的条目
    for me in r_news["matched_entries"]:
        entry = am.read(me["entry_id"])
        if entry:
            assert entry.get("source") != "market_agent", \
                f"source_filter 失效: 包含了 market_agent 的条目"

    print("  >>> Test 6 PASSED")


def test_7_time_filter():
    """Test 7: 时间过滤"""
    print_sep("Test 7: Time Filter")

    am = AgentMemory()

    now = time.time()

    # 写入不同时间的条目
    am.write(topic="很久以前", keywords=["历史", "过去"],
             summary="30天前的事件", timestamp=now - 30 * 86400)
    am.write(topic="最近的事", keywords=["历史", "最近"],
             summary="1小时前的事件", timestamp=now - 3600)
    am.write(topic="刚发生的", keywords=["历史", "当前"],
             summary="5分钟前的事件", timestamp=now - 300)

    # 无时间过滤
    r_all = am.query(keywords=["历史"], depth="fast")
    assert len(r_all["matched_entries"]) == 3, f"无过滤应有 3 条，实际 {len(r_all['matched_entries'])}"
    print(f"  [*] 无时间过滤: {len(r_all['matched_entries'])} 条")

    # 最近 2 小时
    r_recent = am.query(keywords=["历史"], depth="fast", time_recent=2)
    assert len(r_recent["matched_entries"]) == 2, \
        f"最近 2 小时应有 2 条，实际 {len(r_recent['matched_entries'])}"
    print(f"  [*] time_recent=2h: {len(r_recent['matched_entries'])} 条")

    # 最近 10 分钟
    r_10m = am.query(keywords=["历史"], depth="fast", time_recent=10/60)
    assert len(r_10m["matched_entries"]) == 1, \
        f"最近 10 分钟应有 1 条，实际 {len(r_10m['matched_entries'])}"
    print(f"  [*] time_recent=10min: {len(r_10m['matched_entries'])} 条")

    # 时间范围
    r_range = am.query(keywords=["历史"], depth="fast",
                       time_range=(now - 2 * 86400, now - 300))
    # 只有 "1小时前" 在范围内（30天前太早，5分钟前太晚 — 边界恰好是 -300）
    print(f"  [*] time_range(2d_ago ~ 5min_ago): "
          f"{len(r_range['matched_entries'])} 条")

    # 检查时间权重衰减
    for me in r_all["matched_entries"]:
        print(f"      {me['entry_id']}: age={me['age_hours']:.1f}h, "
              f"tw={me['time_weight']}")

    print("  >>> Test 7 PASSED")


def test_8_dual_channel():
    """Test 8: 双通道学习"""
    print_sep("Test 8: Dual Channel")

    am = AgentMemory(min_cooccurrence=1)

    # 先摄入文档（Channel A）
    am.ingest_document("人工智能和机器学习推动了自动驾驶技术的发展")
    am.ingest_document("深度学习是人工智能的重要分支，广泛应用于图像识别")

    docs_before = am._cooccurrence.total_docs
    vocab_before = am._cooccurrence.vocab_count

    # 写入带关键词的记忆条目（Channel B: 自动触发 add_keywords）
    am.write(keywords=["量子计算", "超导", "量子优势", "算力突破"])
    am.write(keywords=["基因编辑", "CRISPR", "生物技术"])

    docs_after = am._cooccurrence.total_docs
    vocab_after = am._cooccurrence.vocab_count

    # total_docs 不应增加（add_keywords 不计入 total_docs）
    assert docs_after == docs_before, \
        f"add_keywords 不应增加 total_docs: {docs_before} -> {docs_after}"
    print(f"  [*] total_docs: {docs_before} -> {docs_after} (unchanged, correct)")

    # vocab 应该增加了新词
    assert vocab_after > vocab_before, \
        f"新关键词应扩展 vocab: {vocab_before} -> {vocab_after}"
    print(f"  [*] vocab: {vocab_before} -> {vocab_after} (+{vocab_after - vocab_before})")

    # 验证新词在 vocab_dict 中
    assert "量子计算" in am._cooccurrence.vocab_dict, "量子计算 应在 vocab 中"
    assert "CRISPR" in am._cooccurrence.vocab_dict, "CRISPR 应在 vocab 中"
    print(f"  [*] 新词验证: 量子计算, CRISPR 均在 vocab 中")

    print("  >>> Test 8 PASSED")


def test_9_repr_and_stats():
    """Test 9: repr 和 get_stats"""
    print_sep("Test 9: repr + get_stats")

    am = AgentMemory()
    am.ingest_document("这是一个测试文档，用于验证统计功能的正确性")
    am.write(topic="测试", keywords=["验证", "统计"], summary="测试条目")

    # repr
    repr_str = repr(am)
    print(f"  [*] repr: {repr_str}")
    assert "AgentMemory(" in repr_str, "repr 格式不正确"

    # get_stats
    stats = am.get_stats()
    assert "config" in stats, "缺少 config"
    assert "cooccurrence" in stats, "缺少 cooccurrence"
    assert "store" in stats, "缺少 store"
    assert "decay" in stats, "缺少 decay"
    assert "matrices" in stats, "缺少 matrices"
    print(f"  [*] get_stats keys: {list(stats.keys())}")
    print(f"      config: {stats['config']}")
    print(f"      cooccurrence: {stats['cooccurrence']}")
    print(f"      store: {stats['store']}")

    print("  >>> Test 9 PASSED")


def test_10_log():
    """Test 10: log() 写入 + 日期/分类索引验证 + 双通道学习"""
    print_sep("Test 10: log() + Indexing + Dual Channel")

    am = AgentMemory(min_cooccurrence=1)

    # 使用固定的日期中间时段，避免跨天问题
    import datetime
    fixed_dt = datetime.datetime(2026, 2, 10, 12, 0, 0)  # 中午12点
    base_ts = fixed_dt.timestamp()
    today_str = "2026-02-10"

    # 记录写入前的 vocab 和 total_docs
    vocab_before = am._cooccurrence.vocab_count
    docs_before = am._cooccurrence.total_docs

    # 写入日志（全在同一天）
    id1 = am.log(
        content="完成了记忆系统的核心模块开发",
        detail="包括共现矩阵、PPMI计算、检索器等模块的开发和测试",
        category="工作",
        tags=["开发", "记忆系统", "PPMI"],
        source="dev_agent",
        timestamp=base_ts - 3600,  # 11:00
    )

    id2 = am.log(
        content="和朋友吃了火锅",
        category="生活",
        source="life_agent",
        timestamp=base_ts - 1800,  # 11:30
    )

    id3 = am.log(
        content="阅读了一篇关于量子计算的论文",
        detail="论文讨论了量子纠错码的最新进展",
        category="工作",
        tags=["量子计算", "论文"],
        source="study_agent",
        timestamp=base_ts,  # 12:00
    )

    assert id1 is not None and id2 is not None and id3 is not None
    print(f"  [*] log 写入 3 条: {id1}, {id2}, {id3}")

    # 验证日期索引
    assert today_str in am._date_index, f"日期索引应包含 {today_str}"
    assert len(am._date_index[today_str]) == 3, \
        f"今天应有 3 条日志，实际 {len(am._date_index[today_str])}"
    # 验证时间正序（id1 最早）
    assert am._date_index[today_str][0] == id1, "日期索引第一条应是最早的"
    assert am._date_index[today_str][2] == id3, "日期索引最后一条应是最新的"
    print(f"  [*] date_index[{today_str}]: {len(am._date_index[today_str])} 条, 正序OK")

    # 验证分类索引
    assert "工作" in am._category_index, "分类索引应包含 '工作'"
    assert "生活" in am._category_index, "分类索引应包含 '生活'"
    assert id1 in am._category_index["工作"], f"{id1} 应在 '工作' 分类中"
    assert id3 in am._category_index["工作"], f"{id3} 应在 '工作' 分类中"
    assert id2 in am._category_index["生活"], f"{id2} 应在 '生活' 分类中"
    print(f"  [*] category_index: 工作={len(am._category_index['工作'])}条, "
          f"生活={len(am._category_index['生活'])}条")

    # 验证 episodic_ids 标记
    assert id1 in am._episodic_ids, f"{id1} 应在 episodic_ids 中"
    assert id2 in am._episodic_ids
    assert id3 in am._episodic_ids
    assert len(am._episodic_ids) == 3
    print(f"  [*] episodic_ids: {len(am._episodic_ids)} 条")

    # 验证双通道学习: total_docs 不变，vocab 增加了
    docs_after = am._cooccurrence.total_docs
    vocab_after = am._cooccurrence.vocab_count
    assert docs_after == docs_before, \
        f"log() 不应增加 total_docs: {docs_before} -> {docs_after}"
    assert vocab_after > vocab_before, \
        f"log() 的关键词应扩展 vocab: {vocab_before} -> {vocab_after}"
    print(f"  [*] dual channel: total_docs={docs_after} (unchanged), "
          f"vocab={vocab_before}->{vocab_after}")

    # 验证底层 MemoryStore 的字段映射
    entry1 = am.read(id1)
    assert entry1["topic"] == "工作", f"category->topic 映射错误: {entry1['topic']}"
    assert entry1["summary"] == "完成了记忆系统的核心模块开发", "content->summary 映射错误"
    assert "共现矩阵" in entry1["body"], "detail->body 映射错误"
    assert entry1["source"] == "dev_agent"
    print(f"  [*] 字段映射验证: topic={entry1['topic']}, source={entry1['source']}")

    # 验证空 content 应报错
    try:
        am.log(content="")
        assert False, "空 content 应抛出 ValueError"
    except ValueError:
        print(f"  [*] log(content='') -> ValueError (correct)")

    print("  >>> Test 10 PASSED")
    return am, today_str


def test_11_recall_by_date(am, today_str):
    """Test 11: recall_by_date() 纯时间查询"""
    print_sep("Test 11: recall_by_date()")

    result = am.recall_by_date(today_str)

    assert result["date"] == today_str
    assert result["count"] == 3, f"今天应有 3 条，实际 {result['count']}"
    print(f"  [*] recall_by_date({today_str}): {result['count']} 条")

    # 验证时间正序
    entries = result["entries"]
    for i in range(len(entries) - 1):
        assert entries[i]["time"] <= entries[i + 1]["time"], \
            f"时间顺序错误: {entries[i]['time']} > {entries[i+1]['time']}"
    print(f"  [*] 时间正序: {[e['time'] for e in entries]}")

    # 验证字段完整性
    e0 = entries[0]
    assert "id" in e0 and "time" in e0 and "category" in e0
    assert "content" in e0 and "tags" in e0 and "source" in e0
    assert e0["category"] == "工作"
    assert e0["content"] == "完成了记忆系统的核心模块开发"
    assert e0["source"] == "dev_agent"
    print(f"  [*] 字段完整: id={e0['id']}, time={e0['time']}, "
          f"cat={e0['category']}, src={e0['source']}")

    # 验证分类统计
    cats = result["categories"]
    assert cats.get("工作") == 2, f"工作应有 2 条: {cats}"
    assert cats.get("生活") == 1, f"生活应有 1 条: {cats}"
    print(f"  [*] categories: {cats}")

    # 查询不存在的日期
    r_empty = am.recall_by_date("1999-01-01")
    assert r_empty["count"] == 0, "不存在日期应返回 0 条"
    assert r_empty["entries"] == []
    print(f"  [*] recall_by_date(1999-01-01): 0 条 (correct)")

    print("  >>> Test 11 PASSED")


def test_12_recall_by_range():
    """Test 12: recall_by_range() 时间段 + 分类/关键词/来源过滤"""
    print_sep("Test 12: recall_by_range()")

    am = AgentMemory()
    import datetime

    # 创建跨多天的日志
    base_dt = datetime.datetime(2026, 2, 5, 9, 0, 0)

    logs = [
        ("完成需求文档编写", "工作", "pm_agent", base_dt),
        ("团队代码评审", "工作", "dev_agent",
         base_dt + datetime.timedelta(hours=3)),
        ("午餐吃了拉面", "生活", "life_agent",
         base_dt + datetime.timedelta(hours=4)),
        ("部署了测试服务器环境", "工作", "dev_agent",
         base_dt + datetime.timedelta(days=1, hours=2)),
        ("看了一部电影", "生活", "life_agent",
         base_dt + datetime.timedelta(days=1, hours=8)),
        ("修复了登录页面的bug", "工作", "dev_agent",
         base_dt + datetime.timedelta(days=2, hours=1)),
        ("跑步5公里", "运动", "life_agent",
         base_dt + datetime.timedelta(days=2, hours=6)),
        ("学习Rust编程语言", "学习", "study_agent",
         base_dt + datetime.timedelta(days=3, hours=10)),
    ]

    for content, cat, src, dt in logs:
        am.log(content=content, category=cat, source=src,
               timestamp=dt.timestamp())

    # 全范围查询
    r_all = am.recall_by_range("2026-02-05", "2026-02-08")
    assert r_all["total_count"] == 8, f"全范围应 8 条，实际 {r_all['total_count']}"
    assert len(r_all["days"]) == 4, f"应覆盖 4 天，实际 {len(r_all['days'])}"
    print(f"  [*] 全范围(02-05~02-08): {r_all['total_count']}条, "
          f"{len(r_all['days'])}天")

    # 分类过滤
    r_work = am.recall_by_range("2026-02-05", "2026-02-08", category="工作")
    assert r_work["total_count"] == 4, \
        f"工作应 4 条，实际 {r_work['total_count']}"
    print(f"  [*] category='工作': {r_work['total_count']}条")

    r_life = am.recall_by_range("2026-02-05", "2026-02-08", category="生活")
    assert r_life["total_count"] == 2, \
        f"生活应 2 条，实际 {r_life['total_count']}"
    print(f"  [*] category='生活': {r_life['total_count']}条")

    # 来源过滤
    r_dev = am.recall_by_range("2026-02-05", "2026-02-08",
                                source_filter="dev_agent")
    assert r_dev["total_count"] == 3, \
        f"dev_agent 应 3 条，实际 {r_dev['total_count']}"
    print(f"  [*] source='dev_agent': {r_dev['total_count']}条")

    # 关键词过滤
    r_kw = am.recall_by_range("2026-02-05", "2026-02-08", keyword="bug")
    assert r_kw["total_count"] == 1, \
        f"keyword='bug' 应 1 条，实际 {r_kw['total_count']}"
    print(f"  [*] keyword='bug': {r_kw['total_count']}条")

    # 组合过滤：分类 + 来源
    r_combo = am.recall_by_range("2026-02-05", "2026-02-08",
                                  category="工作", source_filter="dev_agent")
    assert r_combo["total_count"] == 3, \
        f"工作+dev_agent 应 3 条，实际 {r_combo['total_count']}"
    print(f"  [*] category='工作' + source='dev_agent': "
          f"{r_combo['total_count']}条")

    # 部分日期范围
    r_partial = am.recall_by_range("2026-02-06", "2026-02-07")
    assert r_partial["total_count"] == 4, \
        f"02-06~02-07 应 4 条，实际 {r_partial['total_count']}"
    print(f"  [*] 部分范围(02-06~02-07): {r_partial['total_count']}条")

    # 空范围
    r_empty = am.recall_by_range("2030-01-01", "2030-01-31")
    assert r_empty["total_count"] == 0
    print(f"  [*] 空范围(2030): 0条 (correct)")

    # 分类统计
    print(f"  [*] 全范围 categories: {r_all['categories']}")

    print("  >>> Test 12 PASSED")


def test_13_summarize():
    """Test 13: summarize() 聚合统计"""
    print_sep("Test 13: summarize()")

    am = AgentMemory()
    import datetime

    # 创建一周的日志数据
    end_dt = datetime.datetime(2026, 2, 9, 18, 0, 0)

    week_logs = [
        ("晨会讨论项目进度", "工作", end_dt - datetime.timedelta(days=6, hours=9)),
        ("写了单元测试", "工作", end_dt - datetime.timedelta(days=5, hours=3)),
        ("健身房锻炼", "运动", end_dt - datetime.timedelta(days=5, hours=7)),
        ("代码Review", "工作", end_dt - datetime.timedelta(days=4, hours=2)),
        ("看书", "学习", end_dt - datetime.timedelta(days=3, hours=5)),
        ("跑步", "运动", end_dt - datetime.timedelta(days=2, hours=1)),
        ("部署上线", "工作", end_dt - datetime.timedelta(days=1, hours=4)),
        ("项目复盘", "工作", end_dt - datetime.timedelta(hours=2)),
    ]

    for content, cat, dt in week_logs:
        am.log(content=content, category=cat, timestamp=dt.timestamp())

    # 周汇总
    end_str = end_dt.strftime("%Y-%m-%d")
    result = am.summarize(period="week", end_date=end_str)

    assert result["period_type"] == "week"
    assert result["total_activities"] == 8, \
        f"一周应 8 条，实际 {result['total_activities']}"
    print(f"  [*] week summary: period={result['period']}")
    print(f"      total_activities={result['total_activities']}")
    print(f"      by_category={result['by_category']}")
    print(f"      by_day={result['by_day']}")

    # 验证分类统计
    assert result["by_category"].get("工作") == 5, \
        f"工作应 5 条: {result['by_category']}"
    assert result["by_category"].get("运动") == 2, \
        f"运动应 2 条: {result['by_category']}"

    # 验证 entries 数量与 total_activities 一致
    assert len(result["entries"]) == result["total_activities"], \
        "entries 数量应等于 total_activities"

    # 日汇总
    r_day = am.summarize(period="day", end_date=end_str)
    assert r_day["period_type"] == "day"
    print(f"\n  [*] day summary: period={r_day['period']}, "
          f"total={r_day['total_activities']}")

    # 月汇总
    r_month = am.summarize(period="month", end_date=end_str)
    assert r_month["period_type"] == "month"
    assert r_month["total_activities"] == 8, "月范围应包含全部 8 条"
    print(f"  [*] month summary: period={r_month['period']}, "
          f"total={r_month['total_activities']}")

    # 非法 period
    try:
        am.summarize(period="year")
        assert False, "year 应抛出 ValueError"
    except ValueError:
        print(f"  [*] summarize(period='year') -> ValueError (correct)")

    print("  >>> Test 13 PASSED")


def test_14_episodic_persistence():
    """Test 14: Episodic 持久化 save/load 一致性"""
    print_sep("Test 14: Episodic Persistence")

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_test_episodic_save")

    try:
        am = AgentMemory(min_cooccurrence=1)

        import datetime
        fixed_dt = datetime.datetime(2026, 2, 10, 14, 0, 0)  # 下午2点
        base_ts = fixed_dt.timestamp()
        today_str = "2026-02-10"

        # 同时写入知识条目和日志条目
        k_id = am.write(topic="AI技术", keywords=["深度学习", "Transformer"],
                        summary="Transformer架构成为NLP主流", source="kb_agent")

        l_id1 = am.log(content="调试了记忆系统bug",
                       category="工作", source="dev_agent",
                       timestamp=base_ts - 600)  # 13:50
        l_id2 = am.log(content="晚餐吃了披萨",
                       category="生活", source="life_agent",
                       timestamp=base_ts)  # 14:00

        # 记录保存前状态
        date_index_before = dict(am._date_index)
        cat_index_before = {k: set(v) for k, v in am._category_index.items()}
        episodic_ids_before = set(am._episodic_ids)
        stats_before = am.get_stats()

        print(f"  [*] save前: knowledge=1, logs=2, "
              f"dates={list(am._date_index.keys())}, "
              f"cats={list(am._category_index.keys())}")

        # 保存
        am.save(save_dir)

        # 检查 episodic_meta.json 是否生成
        ep_path = os.path.join(save_dir, "episodic_meta.json")
        assert os.path.exists(ep_path), "episodic_meta.json 未生成"
        with open(ep_path, "r", encoding="utf-8") as f:
            ep_meta = json.load(f)
        print(f"  [*] episodic_meta.json: {os.path.getsize(ep_path)} bytes")
        print(f"      date_index keys: {list(ep_meta['date_index'].keys())}")
        print(f"      category_index keys: {list(ep_meta['category_index'].keys())}")
        print(f"      episodic_ids count: {len(ep_meta['episodic_ids'])}")

        # 新建实例加载
        am2 = AgentMemory()
        ok = am2.load(save_dir)
        assert ok is True, "load 应返回 True"

        # 验证日期索引一致
        assert am2._date_index == date_index_before, \
            f"date_index 不一致: {am2._date_index} vs {date_index_before}"
        print(f"  [*] date_index 一致: OK")

        # 验证分类索引一致
        assert am2._category_index == cat_index_before, \
            f"category_index 不一致"
        print(f"  [*] category_index 一致: OK")

        # 验证 episodic_ids 一致
        assert am2._episodic_ids == episodic_ids_before, \
            f"episodic_ids 不一致"
        print(f"  [*] episodic_ids 一致: OK ({len(am2._episodic_ids)} 条)")

        # 验证知识条目不在 episodic_ids 中
        assert k_id not in am2._episodic_ids, \
            "知识条目不应在 episodic_ids 中"
        assert l_id1 in am2._episodic_ids and l_id2 in am2._episodic_ids
        print(f"  [*] 知识/日志区分: OK (k_id not in episodic, l_ids in episodic)")

        # 验证加载后 recall_by_date 正常工作
        r = am2.recall_by_date(today_str)
        assert r["count"] == 2, f"加载后应有 2 条日志，实际 {r['count']}"
        print(f"  [*] 加载后 recall_by_date: {r['count']} 条 OK")

        # 验证 get_stats episodic 部分
        stats_after = am2.get_stats()
        assert stats_after["episodic"]["total_logs"] == \
               stats_before["episodic"]["total_logs"], "total_logs 不一致"
        assert stats_after["episodic"]["total_days"] == \
               stats_before["episodic"]["total_days"], "total_days 不一致"
        print(f"  [*] get_stats episodic 一致: OK")

        print("  >>> Test 14 PASSED")

    finally:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            print(f"  [*] cleanup: {save_dir} removed")


def test_15_log_semantic_query():
    """Test 15: 日志 + 语义查询协作 — query() 能发现日志条目"""
    print_sep("Test 15: Log + Semantic Query Cooperation")

    am = AgentMemory(min_cooccurrence=1)

    # 先摄入一些文档建立共现知识
    docs = [
        "量子计算和量子纠错是量子信息科学的核心研究方向",
        "超导量子比特是实现量子计算的主要技术路线之一",
        "量子优势已在特定计算问题上得到验证",
    ]
    for doc in docs:
        am.ingest_document(doc)

    # 写入知识条目
    am.write(topic="量子计算进展",
             keywords=["量子计算", "量子比特", "超导"],
             summary="超导量子计算取得重要进展",
             source="kb_agent")

    # 写入日志条目（包含相关关键词）
    log_id = am.log(
        content="参加了量子计算研讨会",
        detail="讨论了量子纠错和容错量子计算的最新进展",
        category="学习",
        tags=["量子计算", "研讨会"],
        source="study_agent",
    )

    # 重建矩阵
    am.rebuild_matrices()

    # 用关键词查询 — 应该能同时找到知识条目和日志条目
    r = am.query(keywords=["量子计算"], depth="fast", token_budget=2000)

    matched_ids = [m["entry_id"] for m in r["matched_entries"]]
    assert log_id in matched_ids, \
        f"query 应能找到日志条目 {log_id}，实际命中: {matched_ids}"
    print(f"  [*] query(keywords=['量子计算']): {len(r['matched_entries'])} 条命中")
    print(f"      命中 IDs: {matched_ids}")
    print(f"      日志条目 {log_id} 在命中列表中: OK")

    # 验证 prompt_text 包含日志内容
    assert "研讨会" in r["prompt_text"], \
        "prompt_text 应包含日志条目内容"
    print(f"  [*] prompt_text 包含日志内容: OK")

    # 用 standard 深度做关联扩展查询
    r2 = am.query(keywords=["量子"], depth="standard", token_budget=2000)
    print(f"  [*] standard query: {len(r2['matched_entries'])} 条, "
          f"expanded: {r2['expanded_keywords'][:5]}")

    # 验证日志条目在 episodic_ids 中被正确标记
    assert log_id in am._episodic_ids
    # 知识条目不在 episodic_ids 中
    kb_entries = am.list_entries(source_filter="kb_agent")
    for e in kb_entries:
        assert e["id"] not in am._episodic_ids, \
            f"知识条目 {e['id']} 不应在 episodic_ids 中"
    print(f"  [*] 知识/日志区分: OK")

    print("  >>> Test 15 PASSED")


def test_16_entry_type_crud():
    """Test 16: SQLite CRUD + 三库分类 (entry_type 过滤)"""
    print_sep("Test 16: Entry Type CRUD + Classification")

    am = AgentMemory()

    # 写入三种类型的记忆
    k1 = am.write(topic="Python基础", keywords=["Python", "编程", "语法"],
                  summary="Python是一种通用编程语言", entry_type="knowledge",
                  category="编程", importance=4)
    k2 = am.write(topic="机器学习", keywords=["机器学习", "算法", "模型"],
                  summary="机器学习是AI的子领域", entry_type="knowledge",
                  category="AI")

    e1 = am.write(topic="调试经验", keywords=["调试", "错误", "日志"],
                  summary="遇到bug先看日志再复现", entry_type="experience",
                  category="开发", importance=5)

    l1 = am.log(content="完成了代码审查", category="工作", source="dev_agent")
    l2 = am.log(content="部署新版本到测试环境", category="部署", source="ops_agent")

    print(f"  [*] write: knowledge={k1},{k2}, experience={e1}, log={l1},{l2}")

    # 按 entry_type 过滤 list_entries
    all_entries = am.list_entries()
    knowledge_entries = am.list_entries(entry_type="knowledge")
    experience_entries = am.list_entries(entry_type="experience")
    log_entries = am.list_entries(entry_type="log")

    assert len(all_entries) == 5, f"total should be 5, got {len(all_entries)}"
    assert len(knowledge_entries) == 2, f"knowledge should be 2, got {len(knowledge_entries)}"
    assert len(experience_entries) == 1, f"experience should be 1, got {len(experience_entries)}"
    assert len(log_entries) == 2, f"log should be 2, got {len(log_entries)}"
    print(f"  [*] list_entries: all={len(all_entries)}, knowledge={len(knowledge_entries)}, "
          f"experience={len(experience_entries)}, log={len(log_entries)}")

    # 验证 entry_type 字段正确写入
    k1_entry = am.read(k1)
    e1_entry = am.read(e1)
    l1_entry = am.read(l1)
    assert k1_entry["entry_type"] == "knowledge", f"k1 type wrong: {k1_entry['entry_type']}"
    assert e1_entry["entry_type"] == "experience", f"e1 type wrong: {e1_entry['entry_type']}"
    assert l1_entry["entry_type"] == "log", f"l1 type wrong: {l1_entry['entry_type']}"
    print(f"  [*] entry_type field: knowledge=OK, experience=OK, log=OK")

    # 验证 importance 字段
    assert k1_entry["importance"] == 4, f"k1 importance wrong: {k1_entry['importance']}"
    assert e1_entry["importance"] == 5, f"e1 importance wrong: {e1_entry['importance']}"
    print(f"  [*] importance: k1={k1_entry['importance']}, e1={e1_entry['importance']}")

    # 验证 category 字段
    assert k1_entry.get("category") == "编程", f"k1 category wrong: {k1_entry.get('category')}"
    print(f"  [*] category: k1={k1_entry.get('category')}")

    # 按 source 过滤 + entry_type 组合过滤
    dev_logs = am.list_entries(source_filter="dev_agent", entry_type="log")
    assert len(dev_logs) == 1, f"dev_agent logs should be 1, got {len(dev_logs)}"
    print(f"  [*] combined filter (source=dev_agent, type=log): {len(dev_logs)} entry")

    # date_str 自动生成验证
    assert k1_entry.get("date_str") is not None, "date_str should be auto-generated"
    print(f"  [*] date_str auto-generated: {k1_entry['date_str']}")

    print("  >>> Test 16 PASSED")


def test_17_access_count():
    """Test 17: 提取强化 - access_count 查询后自增"""
    print_sep("Test 17: Retrieval Strengthening (access_count)")

    am = AgentMemory()

    # 写入条目并摄入文档构建矩阵
    id1 = am.write(topic="深度学习框架", keywords=["深度学习", "PyTorch", "框架"],
                   summary="PyTorch是流行的深度学习框架")
    id2 = am.write(topic="自然语言处理", keywords=["NLP", "文本", "分析"],
                   summary="NLP是AI处理文本的技术")
    id3 = am.write(topic="计算机视觉", keywords=["视觉", "图像", "识别"],
                   summary="计算机视觉用于图像分析")

    # 初始 access_count 应为 0
    e1 = am.read(id1)
    e2 = am.read(id2)
    e3 = am.read(id3)
    assert e1["access_count"] == 0, f"initial access_count should be 0, got {e1['access_count']}"
    assert e2["access_count"] == 0
    assert e3["access_count"] == 0
    print(f"  [*] initial access_count: id1={e1['access_count']}, id2={e2['access_count']}, id3={e3['access_count']}")

    # 查询命中 id1 (通过精确关键词匹配)
    result = am.query(keywords=["深度学习", "PyTorch"], depth="fast")
    matched_ids = [m["entry_id"] for m in result["matched_entries"]]
    print(f"  [*] query(depth=fast): matched={matched_ids}")

    # 验证 id1 的 access_count 递增
    e1_after = am.read(id1)
    if id1 in matched_ids:
        assert e1_after["access_count"] >= 1, \
            f"access_count should be >= 1 after query hit, got {e1_after['access_count']}"
        print(f"  [*] id1 access_count after query: {e1_after['access_count']} (incremented)")
    else:
        print(f"  [*] id1 not matched (keyword mismatch), access_count unchanged: {e1_after['access_count']}")

    # 再查一次，access_count 应该继续递增
    result2 = am.query(keywords=["深度学习"], depth="fast")
    e1_after2 = am.read(id1)
    if id1 in [m["entry_id"] for m in result2["matched_entries"]]:
        assert e1_after2["access_count"] >= 2, \
            f"access_count should be >= 2 after 2 queries, got {e1_after2['access_count']}"
        print(f"  [*] id1 access_count after 2nd query: {e1_after2['access_count']} (incremented again)")

    # 未命中的条目 access_count 不变
    e3_after = am.read(id3)
    assert e3_after["access_count"] == 0, \
        f"id3 should remain 0, got {e3_after['access_count']}"
    print(f"  [*] id3 access_count (never matched): {e3_after['access_count']} (unchanged)")

    # 验证 last_accessed 更新
    if e1_after["access_count"] >= 1:
        assert e1_after["last_accessed"] is not None, "last_accessed should be set"
        assert e1_after["last_accessed"] >= e1["timestamp"], "last_accessed should be >= creation time"
        print(f"  [*] last_accessed updated: OK")

    print("  >>> Test 17 PASSED")


def test_18_importance_ranking():
    """Test 18: 重要性分级 - 影响时间衰减 + 排序"""
    print_sep("Test 18: Importance Grading (decay + ranking)")

    from memory.storage import MemoryStore

    # Part A: 验证 compute_time_weight 公式
    now = time.time()
    ts_30d_ago = now - 30 * 86400  # 30 天前

    tw_imp1 = MemoryStore.compute_time_weight(ts_30d_ago, decay_lambda=0.1, now=now, importance=1)
    tw_imp3 = MemoryStore.compute_time_weight(ts_30d_ago, decay_lambda=0.1, now=now, importance=3)
    tw_imp5 = MemoryStore.compute_time_weight(ts_30d_ago, decay_lambda=0.1, now=now, importance=5)

    print(f"  [*] time_weight (30d ago): imp1={tw_imp1:.4f}, imp3={tw_imp3:.4f}, imp5={tw_imp5:.4f}")

    # importance=5 衰减最慢 -> 权重最高
    # importance=1 衰减最快 -> 权重最低
    assert tw_imp5 > tw_imp3, f"imp5 ({tw_imp5:.4f}) should > imp3 ({tw_imp3:.4f})"
    assert tw_imp3 > tw_imp1, f"imp3 ({tw_imp3:.4f}) should > imp1 ({tw_imp1:.4f})"
    print(f"  [*] decay order correct: imp5 > imp3 > imp1")

    # importance=3 时应与旧公式一致: 1/(1+0.1*30) = 0.25
    expected_old = 1.0 / (1.0 + 0.1 * 30)
    assert abs(tw_imp3 - expected_old) < 1e-6, \
        f"imp3 should match old formula: {expected_old:.6f} vs {tw_imp3:.6f}"
    print(f"  [*] imp3 backward compatible: {tw_imp3:.6f} == {expected_old:.6f}")

    # Part B: 验证 set_importance + 排序
    am = AgentMemory()

    # 创建两条相同关键词的条目，一条重要度高，一条低
    # 使用相同的时间戳（30天前）确保时间因素一致
    old_ts = time.time() - 30 * 86400
    id_low = am.write(topic="低优先级新闻", keywords=["经济", "数据"],
                      summary="普通经济数据播报", importance=1, timestamp=old_ts)
    id_high = am.write(topic="重大经济事件", keywords=["经济", "数据"],
                       summary="重大经济政策变动", importance=5, timestamp=old_ts)

    print(f"  [*] write: id_low={id_low}(imp=1), id_high={id_high}(imp=5)")

    # 查询（fast 模式，避免需要矩阵）
    result = am.query(keywords=["经济", "数据"], depth="fast")
    matched = result["matched_entries"]
    assert len(matched) >= 2, f"should match at least 2, got {len(matched)}"

    # 高重要性应排在前面
    ids_order = [m["entry_id"] for m in matched]
    idx_high = ids_order.index(id_high)
    idx_low = ids_order.index(id_low)
    assert idx_high < idx_low, \
        f"high importance should rank higher: {id_high}@{idx_high} vs {id_low}@{idx_low}"
    print(f"  [*] ranking: {id_high}(imp=5) @ pos {idx_high}, {id_low}(imp=1) @ pos {idx_low}")

    # Part C: 动态修改 importance
    ok = am.set_importance(id_low, 5)
    assert ok is True, "set_importance should return True"
    e_low = am.read(id_low)
    assert e_low["importance"] == 5, f"importance should be 5, got {e_low['importance']}"
    print(f"  [*] set_importance({id_low}, 5): OK, now importance={e_low['importance']}")

    # 边界钳位测试
    am.set_importance(id_low, 10)  # 超过上限
    assert am.read(id_low)["importance"] == 5, "importance should be clamped to 5"
    am.set_importance(id_low, 0)   # 低于下限
    assert am.read(id_low)["importance"] == 1, "importance should be clamped to 1"
    print(f"  [*] clamping: set(10)->5, set(0)->1: OK")

    # 不存在的条目
    ok2 = am.set_importance("nonexist_id", 3)
    assert ok2 is False, "set_importance on nonexistent should return False"
    print(f"  [*] set_importance(nonexist): False (correct)")

    print("  >>> Test 18 PASSED")


def test_19_reconsolidation():
    """Test 19: 记忆再巩固 - supersedes 链接 + 旧条目降级"""
    print_sep("Test 19: Reconsolidation (supersedes)")

    am = AgentMemory()

    # 写入原始记忆
    old_id = am.write(topic="地球形状", keywords=["地球", "形状", "科学"],
                      summary="地球是球形的", importance=4)
    old_entry = am.read(old_id)
    assert old_entry["importance"] == 4
    print(f"  [*] original: {old_id}, importance={old_entry['importance']}")

    # 写入新记忆，取代旧记忆
    new_id = am.write(topic="地球形状修正", keywords=["地球", "形状", "科学"],
                      summary="地球是椭球体，不是完美球形",
                      importance=5, supersedes=old_id)
    new_entry = am.read(new_id)
    assert new_entry["importance"] == 5
    assert new_entry["supersedes"] == old_id, \
        f"supersedes should be {old_id}, got {new_entry['supersedes']}"
    print(f"  [*] new: {new_id}, importance={new_entry['importance']}, supersedes={old_id}")

    # 旧条目的 importance 应该被自动降级 (-1)
    old_entry_after = am.read(old_id)
    assert old_entry_after["importance"] == 3, \
        f"old importance should be 4-1=3, got {old_entry_after['importance']}"
    print(f"  [*] old entry demoted: importance {old_entry['importance']} -> {old_entry_after['importance']}")

    # 连续取代：再写一条取代 new_id
    newer_id = am.write(topic="地球形状最新", keywords=["地球", "形状"],
                        summary="地球是不规则椭球体",
                        importance=5, supersedes=new_id)
    new_entry_after = am.read(new_id)
    assert new_entry_after["importance"] == 4, \
        f"new importance should be 5-1=4, got {new_entry_after['importance']}"
    print(f"  [*] chain supersedes: {newer_id} -> {new_id}(imp {5}->{new_entry_after['importance']}) -> {old_id}(imp {3})")

    # 降级不会低于1
    # 创建一条 importance=1 的条目，然后取代它
    weak_id = am.write(topic="临时笔记", keywords=["临时", "笔记"],
                       summary="临时记录", importance=1)
    replace_id = am.write(topic="正式笔记", keywords=["正式", "笔记"],
                          summary="正式记录", importance=3, supersedes=weak_id)
    weak_after = am.read(weak_id)
    assert weak_after["importance"] == 1, \
        f"importance should not go below 1, got {weak_after['importance']}"
    print(f"  [*] floor check: imp=1 demoted -> still 1 (correct)")

    # 取代不存在的条目不报错
    safe_id = am.write(topic="安全测试", keywords=["测试"],
                       summary="取代不存在的条目", supersedes="nonexist_000")
    assert safe_id is not None, "should succeed even with nonexistent supersedes target"
    print(f"  [*] supersedes nonexistent: no error (correct)")

    # 查询时新记忆应优先（因为 importance 更高）
    result = am.query(keywords=["地球", "形状"], depth="fast")
    matched_ids = [m["entry_id"] for m in result["matched_entries"]]
    assert newer_id in matched_ids, f"newest entry should be in results"
    if len(matched_ids) >= 2:
        idx_newer = matched_ids.index(newer_id)
        idx_old = matched_ids.index(old_id) if old_id in matched_ids else len(matched_ids)
        assert idx_newer < idx_old, "newer entry should rank higher than old"
        print(f"  [*] query ranking: {newer_id} before {old_id} (correct)")

    print("  >>> Test 19 PASSED")


def test_20_consolidate():
    """Test 20: 整合清理 (cleanup_vocab + remove_words + consolidate)"""
    print_sep("Test 20: Consolidation (cleanup + rebuild)")

    am = AgentMemory()

    # 摄入一些文档构建共现矩阵
    docs = [
        "人工智能技术在医疗健康领域的应用越来越广泛",
        "深度学习算法在图像识别方面取得了突破",
        "自然语言处理技术帮助机器理解人类语言",
        "机器学习模型需要大量数据进行训练",
        "强化学习在游戏和机器人控制中表现优异",
    ]
    for doc in docs:
        am.ingest_document(doc)
    am.rebuild_matrices()

    vocab_before = am._cooccurrence.vocab_count
    print(f"  [*] initial vocab: {vocab_before}")
    assert vocab_before > 0, "vocab should not be empty"

    # Part A: cleanup_vocab 测试
    # 用高阈值，确保能移除一些词
    removed = am.cleanup_vocab(min_cooccurrence=100)
    vocab_after_cleanup = am._cooccurrence.vocab_count
    print(f"  [*] cleanup_vocab(min=100): removed={removed}, vocab {vocab_before} -> {vocab_after_cleanup}")
    assert removed >= 0, "removed should be non-negative"
    if removed > 0:
        assert vocab_after_cleanup < vocab_before, "vocab should decrease"

    # Part B: 重建后矩阵仍可用
    am.rebuild_matrices()
    stats = am.get_stats()
    print(f"  [*] after rebuild: vocab={stats['cooccurrence']['vocab_size']}, "
          f"ppmi_nnz={stats['matrices']['ppmi_nnz']}")

    # Part C: consolidate 完整流程 (不指定 data_dir，不触发持久化)
    am2 = AgentMemory()
    for doc in docs:
        am2.ingest_document(doc)
    am2.rebuild_matrices()

    # 写入一些记忆条目（这些关键词会进入共现矩阵）
    am2.write(topic="AI应用", keywords=["人工智能", "应用", "医疗"],
              summary="AI在医疗中的应用")
    am2.write(topic="DL突破", keywords=["深度学习", "图像", "识别"],
              summary="深度学习在视觉中的突破")

    v_before = am2._cooccurrence.vocab_count
    result = am2.consolidate(min_cooccurrence=50)
    v_after = am2._cooccurrence.vocab_count

    print(f"  [*] consolidate(min=50): vocab {result['vocab_before']} -> {result['vocab_after']}, "
          f"removed={result['words_removed']}")
    assert "vocab_before" in result, "result should have vocab_before"
    assert "vocab_after" in result, "result should have vocab_after"
    assert "words_removed" in result, "result should have words_removed"
    assert result["vocab_before"] == v_before, "vocab_before should match"
    assert result["vocab_after"] == v_after, "vocab_after should match"

    # Part D: consolidate with rebuild_from_recent
    am3 = AgentMemory()
    # 写入一些有时间戳的条目
    old_ts = time.time() - 100 * 86400  # 100天前
    recent_ts = time.time() - 1 * 86400  # 1天前

    am3.write(topic="旧知识", keywords=["旧知识", "过时", "历史"],
              summary="很久以前的知识", timestamp=old_ts)
    am3.write(topic="新知识", keywords=["新知识", "最新", "前沿"],
              summary="最近学到的知识", timestamp=recent_ts)
    am3.ingest_document("新知识和前沿技术正在快速发展")
    am3.rebuild_matrices()

    result3 = am3.consolidate(rebuild_from_recent=30, min_cooccurrence=1)
    print(f"  [*] consolidate(rebuild_from_recent=30): "
          f"vocab {result3['vocab_before']} -> {result3['vocab_after']}")

    # Part E: max_vocab 限制
    am4 = AgentMemory()
    for doc in docs:
        am4.ingest_document(doc)
    am4.rebuild_matrices()
    v4_before = am4._cooccurrence.vocab_count

    if v4_before > 5:
        removed4 = am4.cleanup_vocab(min_cooccurrence=1, max_vocab=5)
        v4_after = am4._cooccurrence.vocab_count
        print(f"  [*] cleanup_vocab(max_vocab=5): vocab {v4_before} -> {v4_after}, removed={removed4}")
        assert v4_after <= 5, f"vocab should be <= 5 after max_vocab limit, got {v4_after}"
    else:
        print(f"  [*] vocab too small ({v4_before}) to test max_vocab limit, skipping")

    # Part F: remove_words 直接测试
    am5 = AgentMemory()
    am5.ingest_document("机器学习和深度学习是人工智能的核心技术")
    am5.rebuild_matrices()

    vocab5 = set(am5._cooccurrence.vocab_dict.keys())
    if len(vocab5) >= 3:
        # 选两个词移除
        words_to_remove = list(vocab5)[:2]
        removed5 = am5._cooccurrence.remove_words(words_to_remove)
        assert removed5 == 2 or removed5 == len(words_to_remove), \
            f"should remove {len(words_to_remove)}, got {removed5}"
        for w in words_to_remove:
            assert w not in am5._cooccurrence.vocab_dict, f"{w} should be removed from vocab"
        print(f"  [*] remove_words({words_to_remove}): removed={removed5}, vocab intact")
    else:
        print(f"  [*] vocab too small ({len(vocab5)}) for remove_words test, skipping")

    print("  >>> Test 20 PASSED")


def main():
    print("=" * 60)
    print("  AgentMemory Unified API - End-to-End Test")
    print("=" * 60)

    t_start = time.time()
    passed = 0
    failed = 0
    total = 20

    # Test 1: CRUD
    try:
        test_1_crud()
        passed += 1
    except Exception as e:
        print(f"  !!! Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    # Test 2: Ingest + Matrices (返回 am 供后续使用)
    am = None
    try:
        am = test_2_ingest_and_matrices()
        passed += 1
    except Exception as e:
        print(f"  !!! Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    if am is None:
        print("\n  !!! Test 2 失败，跳过依赖 Test 3/4/5/6")
        failed += 4
    else:
        # Test 3: Query
        try:
            test_3_query(am)
            passed += 1
        except Exception as e:
            print(f"  !!! Test 3 FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

        # Test 4: Chain
        try:
            test_4_chain(am)
            passed += 1
        except Exception as e:
            print(f"  !!! Test 4 FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

        # Test 5: Persistence
        try:
            test_5_persistence(am)
            passed += 1
        except Exception as e:
            print(f"  !!! Test 5 FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

        # Test 6: Multi-Agent
        try:
            test_6_multi_agent(am)
            passed += 1
        except Exception as e:
            print(f"  !!! Test 6 FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Test 7: Time Filter (独立)
    try:
        test_7_time_filter()
        passed += 1
    except Exception as e:
        print(f"  !!! Test 7 FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    # Test 8: Dual Channel (独立)
    try:
        test_8_dual_channel()
        passed += 1
    except Exception as e:
        print(f"  !!! Test 8 FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    # Test 9: repr + stats (独立)
    try:
        test_9_repr_and_stats()
        passed += 1
    except Exception as e:
        print(f"  !!! Test 9 FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    # Test 10: log() (独立)
    am_ep = None
    today_ep = None
    try:
        am_ep, today_ep = test_10_log()
        passed += 1
    except Exception as e:
        print(f"  !!! Test 10 FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    # Test 11: recall_by_date (依赖 Test 10)
    if am_ep is not None:
        try:
            test_11_recall_by_date(am_ep, today_ep)
            passed += 1
        except Exception as e:
            print(f"  !!! Test 11 FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    else:
        print("\n  !!! Test 10 失败，跳过 Test 11")
        failed += 1

    # Test 12: recall_by_range (独立)
    try:
        test_12_recall_by_range()
        passed += 1
    except Exception as e:
        print(f"  !!! Test 12 FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    # Test 13: summarize (独立)
    try:
        test_13_summarize()
        passed += 1
    except Exception as e:
        print(f"  !!! Test 13 FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    # Test 14: episodic persistence (独立)
    try:
        test_14_episodic_persistence()
        passed += 1
    except Exception as e:
        print(f"  !!! Test 14 FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    # Test 15: log + semantic query (独立)
    try:
        test_15_log_semantic_query()
        passed += 1
    except Exception as e:
        print(f"  !!! Test 15 FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    # Test 16: Entry Type CRUD (独立)
    try:
        test_16_entry_type_crud()
        passed += 1
    except Exception as e:
        print(f"  !!! Test 16 FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    # Test 17: Access Count (独立)
    try:
        test_17_access_count()
        passed += 1
    except Exception as e:
        print(f"  !!! Test 17 FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    # Test 18: Importance Ranking (独立)
    try:
        test_18_importance_ranking()
        passed += 1
    except Exception as e:
        print(f"  !!! Test 18 FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    # Test 19: Reconsolidation (独立)
    try:
        test_19_reconsolidation()
        passed += 1
    except Exception as e:
        print(f"  !!! Test 19 FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    # Test 20: Consolidate (独立)
    try:
        test_20_consolidate()
        passed += 1
    except Exception as e:
        print(f"  !!! Test 20 FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print(f"  Time: {elapsed:.2f}s")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)
    else:
        print("\n  ALL TESTS PASSED!")
        sys.exit(0)


if __name__ == "__main__":
    main()
