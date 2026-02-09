# Agent Memory Tool

**AI Agent Long-term Associative Memory System**
**AI 智能体长期关联记忆系统**

A brain-inspired passive memory middleware for AI agents. No vector database, no GPU required -- pure Python with sparse matrix math, delivering human-like memory association, retrieval strengthening, and knowledge consolidation.

一个受人脑启发的被动记忆中间件，为 AI 智能体提供长期关联记忆能力。无需向量数据库，无需 GPU -- 纯 Python 稀疏矩阵计算，实现类人的记忆关联、提取强化和知识整合。

---

## Why This Project? / 为什么做这个项目？

Current AI agents lose all context between sessions. Vector databases retrieve by semantic similarity but miss **logical connections** -- they can't discover that "Trump's tariffs" and "A-share market crash" are linked through "China-US trade -> export decline -> market panic".

当前 AI 智能体在会话之间会丢失所有上下文。向量数据库通过语义相似度检索，但忽略了**逻辑关联** -- 它们无法发现 "特朗普关税" 和 "A股暴跌" 之间通过 "中美贸易 -> 出口下降 -> 市场恐慌" 的隐藏链条。

This system uses **co-occurrence matrices + PPMI + graph reasoning** to build associative links between concepts, just like how human memory works -- not by filing things into folders, but by strengthening connections between ideas that appear together.

本系统使用**共现矩阵 + PPMI + 图推理**在概念之间建立关联链接，就像人脑记忆的运作方式 -- 不是把东西归档到文件夹，而是加强一起出现的概念之间的联系。

---

## Architecture / 架构

```
┌─────────────────────────────────────────────────────────────┐
│                     AI Agent (consumer)                     │
│              Reads memory / Writes memory                   │
├─────────────────────────────────────────────────────────────┤
│                    AgentMemory (unified API)                 │
│                                                             │
│  Layer 1: Auto-Association Injection (every turn)           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ User input -> keyword extraction -> co-occurrence     │  │
│  │ query -> PPMI expansion -> inverted index retrieval   │  │
│  │ -> layered loading within token budget -> inject      │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  Layer 2: Reasoning Chain Discovery (on-demand)             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Keywords -> PPR + shortest path -> discover hidden    │  │
│  │ intermediate concepts -> attach text evidence         │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  Layer 3: Episodic Memory (activity log)                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Activity logging -> date/category index -> recall     │  │
│  │ -> keywords feed into co-occurrence matrix            │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   SQLite     │  Co-occur    │   PPMI       │   Inverted     │
│   Storage    │  Matrix      │   Matrix     │   Index        │
│   (entries)  │  (sparse)    │   (sparse)   │   (keywords)   │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

### Three Memory Types / 三种记忆类型

| Type | Description | Example |
|------|-------------|---------|
| `knowledge` | Facts, concepts, stable knowledge / 事实、概念、稳定知识 | "Python is a programming language" |
| `experience` | Lessons learned, patterns / 经验教训、模式总结 | "Always check logs before debugging" |
| `log` | Activity events / 活动事件记录 | "Completed code review at 14:00" |

### Three Brain-Inspired Mechanisms / 三大脑启发机制

| Mechanism | How it works |
|-----------|-------------|
| **Retrieval Strengthening** / 提取强化 | `access_count` increments on every query hit. Frequently recalled memories stay active. / 每次查询命中自动递增 `access_count`，被频繁回忆的记忆保持活跃 |
| **Importance Grading** / 重要度分级 | `importance` (1-5) affects time decay rate and ranking weight. Important memories fade slower. / `importance` (1-5) 影响时间衰减速率和排名权重，重要记忆遗忘更慢 |
| **Reconsolidation** / 记忆再巩固 | `supersedes` links new memory to old; old memory's importance auto-decrements. / `supersedes` 将新记忆链接到旧记忆，旧记忆重要度自动降级 |

---

## Quick Start / 快速开始

### Installation / 安装

```bash
# Clone
git clone https://github.com/lihua179/agent_memory_tool.git
cd agent_memory_tool

# Install dependencies / 安装依赖
pip install numpy scipy jieba networkx tiktoken
```

### Basic Usage / 基本用法

```python
from memory import AgentMemory

# Initialize (in-memory mode) / 初始化（内存模式）
am = AgentMemory()

# Or with SQLite persistence / 或使用 SQLite 持久化
# am = AgentMemory(data_dir="./memory_data")

# 1. Ingest documents to build knowledge graph / 摄入文档构建知识图谱
am.ingest_document("Trump announced new tariffs on Chinese goods, impacting bilateral trade.")
am.ingest_document("The A-share market dropped sharply as export companies reported losses.")
am.rebuild_matrices()

# 2. Write structured memories / 写入结构化记忆
am.write(
    topic="US-China Trade Friction",
    keywords=["tariffs", "trade", "China", "exports"],
    summary="Escalating trade tensions between US and China",
    importance=4
)

# 3. Query -- auto-injects relevant context into prompt / 查询 -- 自动注入相关上下文
result = am.query(keywords=["tariffs", "exports"], depth="standard", token_budget=1000)
print(result["prompt_text"])      # Ready to inject into agent prompt
print(result["matched_entries"])  # Matched memory entries
print(result["expanded_keywords"])  # PPMI-expanded related terms

# 4. Discover hidden reasoning chains / 发现隐藏推理链
chain = am.find_chain(["tariffs", "stock market"])
# Might discover: tariffs -> China -> exports -> market -> A-shares
```

### Episodic Memory (Activity Log) / 日志记忆

```python
# Log activities / 记录活动
am.log(content="Fixed critical bug in auth module", category="coding", source="dev_agent")
am.log(content="Deployed v2.1 to production", category="deployment", source="ops_agent")

# Recall by date / 按日期回忆
entries = am.recall_by_date("2026-02-10")

# Recall by range with filters / 按时间段+过滤条件回忆
entries = am.recall_by_range("2026-02-01", "2026-02-10", category="coding")

# Weekly summary / 周报汇总
summary = am.summarize(period="week")
```

### Memory Consolidation / 记忆整合

```python
# Update importance / 修改重要度
am.set_importance(entry_id, level=5)

# Reconsolidate -- new knowledge supersedes old / 再巩固 -- 新知识取代旧知识
new_id = am.write(
    topic="Updated finding",
    keywords=["earth", "shape"],
    summary="Earth is an oblate spheroid, not a perfect sphere",
    importance=5,
    supersedes=old_entry_id   # Old entry's importance auto-decrements
)

# Sleep consolidation -- prune low-frequency vocab, rebuild matrices
# 睡眠整理 -- 剪枝低频词汇，重建矩阵
result = am.consolidate(min_cooccurrence=3, rebuild_from_recent=90)
```

### Persistence / 持久化

```python
# Save (matrices + config; entries already in SQLite)
# 保存（矩阵+配置；条目已实时存入 SQLite）
am.save("./memory_data")

# Load / 加载
am.load("./memory_data")
```

---

## Query Depths / 查询深度

| Depth | What it does | Speed |
|-------|-------------|-------|
| `fast` | Exact + fuzzy keyword match via inverted index / 精确+模糊关键词匹配 | ~1ms |
| `standard` | + PPMI association expansion (discovers related terms) / + PPMI 关联扩展 | ~10ms |
| `deep` | + Hidden chain reasoning (PPR + shortest path) / + 隐藏链推理 | ~100ms |

```python
# Fast: direct keyword lookup / 快速：直接关键词查找
result = am.query(keywords=["AI"], depth="fast")

# Standard: expands "AI" to also search "machine learning", "neural network", etc.
# 标准：将 "AI" 扩展为同时搜索 "机器学习"、"神经网络" 等
result = am.query(keywords=["AI"], depth="standard")

# Deep: discovers hidden chains like AI -> healthcare -> diagnosis
# 深度：发现隐藏链条，如 AI -> 医疗 -> 诊断
result = am.query(keywords=["AI", "healthcare"], depth="deep")
```

---

## Scoring Formula / 评分公式

Each matched memory entry is scored by three factors:

每个匹配的记忆条目由三个因素评分：

```
final_score = relevance * time_weight * importance_factor
```

| Factor | Formula | Description |
|--------|---------|-------------|
| `relevance` | keyword match count + fuzzy bonus | How well the entry matches query keywords / 关键词匹配度 |
| `time_weight` | `1 / (1 + adjusted_lambda * age_days)` | Recency decay, importance-aware / 时间衰减（重要度感知） |
| `importance_factor` | `importance / 3.0` | importance=3 is neutral / importance=3 为中性基准 |

**Importance-aware decay / 重要度感知衰减:**

```
adjusted_lambda = decay_lambda * 3.0 / importance

importance=5 -> slower decay (important memories persist)
importance=3 -> standard decay (backward compatible)
importance=1 -> faster decay (trivial memories fade quickly)
```

---

## API Reference / API 参考

### Core Methods / 核心方法

| Method | Description |
|--------|-------------|
| `write(topic, keywords, summary, body, source, entry_type, importance, supersedes)` | Write a memory entry / 写入记忆条目 |
| `read(entry_id)` | Read a single entry / 读取单条记忆 |
| `remove(entry_id)` | Remove an entry / 删除记忆条目 |
| `list_entries(source_filter, entry_type)` | List entries with optional filters / 列出条目（可过滤） |
| `query(keywords, user_input, depth, token_budget, time_recent, time_range, source_filter)` | Multi-level retrieval / 多级检索 |
| `find_chain(anchor_words, top_n, with_evidence)` | Discover hidden reasoning chains / 发现隐藏推理链 |

### Episodic Methods / 日志方法

| Method | Description |
|--------|-------------|
| `log(content, detail, category, tags, source, importance)` | Write an activity log / 写入活动日志 |
| `recall_by_date(date_str)` | Recall activities by date / 按日期回忆 |
| `recall_by_range(start_date, end_date, category, source, keyword)` | Recall by range + filters / 按时间段回忆 |
| `summarize(period)` | Aggregate summary (day/week/month) / 聚合统计 |

### Brain-Inspired Methods / 脑启发方法

| Method | Description |
|--------|-------------|
| `set_importance(entry_id, level)` | Set importance level (1-5) / 设置重要度 |
| `consolidate(min_cooccurrence, max_vocab, rebuild_from_recent)` | Sleep consolidation / 睡眠整理 |
| `cleanup_vocab(min_cooccurrence, max_vocab)` | Prune low-frequency words / 剪枝低频词 |

### Data Methods / 数据方法

| Method | Description |
|--------|-------------|
| `ingest_document(text)` | Ingest a document into co-occurrence matrix / 摄入文档 |
| `ingest_documents_from_csv(csv_path, content_column)` | Batch ingest from CSV / 从 CSV 批量摄入 |
| `rebuild_matrices(min_cooccurrence)` | Rebuild PPMI matrices / 重建 PPMI 矩阵 |
| `save(directory)` | Save matrices + config to disk / 保存矩阵和配置 |
| `load(directory)` | Load from disk / 从磁盘加载 |
| `get_stats()` | Get system statistics / 获取系统统计信息 |

---

## Module Structure / 模块结构

```
agent_memory_tool/
├── memory/
│   ├── __init__.py          # Module entry point / 模块入口
│   ├── agent_memory.py      # Unified API (AgentMemory class) / 统一 API
│   ├── storage.py           # SQLite storage engine / SQLite 存储引擎
│   ├── cooccurrence.py      # Incremental co-occurrence matrix / 增量共现矩阵
│   ├── probability.py       # PPMI & conditional probability / PPMI 和条件概率
│   ├── retriever.py         # Multi-level retriever / 多级检索器
│   ├── chain.py             # Hidden chain reasoning (PPR) / 隐藏链推理
│   ├── inference.py         # Multi-hop inference / 多跳推理
│   └── decay.py             # Time decay manager / 时间衰减管理
├── test_agent_memory.py     # 20 end-to-end tests / 20 个端到端测试
└── README.md
```

---

## Testing / 测试

```bash
python test_agent_memory.py
```

20 tests covering / 20 个测试覆盖:

| # | Test | Description |
|---|------|-------------|
| 1 | CRUD | Basic write/read/remove/list |
| 2 | Ingest + Matrices | Document ingestion + matrix building |
| 3 | Query | Three-level retrieval (fast/standard/deep) |
| 4 | Chain Reasoning | Hidden chain discovery + evidence |
| 5 | Persistence | Save/load + consistency verification |
| 6 | Multi-Agent | Shared memory with source filtering |
| 7 | Time Filter | time_recent / time_range filtering |
| 8 | Dual Channel | Write auto-triggers keyword learning |
| 9 | Stats | repr + get_stats |
| 10 | Log | Activity logging + index verification |
| 11 | Recall by Date | Date-based recall |
| 12 | Recall by Range | Range + category/keyword/source filters |
| 13 | Summarize | Aggregation statistics (day/week/month) |
| 14 | Episodic Persistence | Log save/load consistency |
| 15 | Log + Query Cooperation | Log entries in semantic query results |
| 16 | Entry Type CRUD | Three-store classification (knowledge/experience/log) |
| 17 | Access Count | Retrieval strengthening (access_count increment) |
| 18 | Importance | Importance-aware decay + ranking + clamping |
| 19 | Reconsolidation | Supersedes linking + auto-demotion |
| 20 | Consolidation | cleanup_vocab + remove_words + consolidate flow |

---

## Dependencies / 依赖

| Package | Version | Purpose |
|---------|---------|---------|
| Python | >= 3.9 | Runtime |
| numpy | >= 1.21 | Array operations |
| scipy | >= 1.7 | Sparse matrices |
| jieba | >= 0.42 | Chinese word segmentation |
| networkx | >= 2.6 | Graph algorithms (PPR, shortest path) |
| tiktoken | >= 0.5 | Token counting for budget control |

---

## Design Philosophy / 设计理念

### Why not vector databases? / 为什么不用向量数据库？

Vector databases excel at **semantic similarity** ("find documents about dogs" also returns documents about "puppies"). But they miss **logical associations** -- co-occurrence-based reasoning can discover that "tariffs" and "stock crash" are connected through intermediate concepts like "trade war" and "export decline", even though these words are not semantically similar.

向量数据库擅长**语义相似度**检索，但忽略了**逻辑关联** -- 基于共现的推理能发现 "关税" 和 "股市暴跌" 通过 "贸易战" 和 "出口下降" 等中间概念相连，即使这些词在语义上并不相似。

### Why is the memory system passive? / 为什么记忆系统是被动的？

The database provides three objective knobs: **match score**, **time decay**, and **importance weight**. All subjective decisions -- what is important, what to abstract, what to plan -- are left to the AI agents themselves. The memory system is infrastructure, not intelligence.

数据库提供三个客观旋钮：**匹配分数**、**时间衰减**和**重要度权重**。所有主观决策 -- 什么是重要的、如何抽象、如何规划 -- 都留给 AI 智能体自身决定。记忆系统是基础设施，不是智能。

### Why SQLite? / 为什么用 SQLite？

- Zero configuration, no server needed / 零配置，无需服务器
- Entries persist in real-time (no explicit save needed) / 条目实时持久化
- Single-file database, easy to backup and migrate / 单文件数据库，易于备份迁移
- Good enough for single-agent or moderate multi-agent scenarios / 对单智能体或适度多智能体场景足够好

---

## License

MIT
