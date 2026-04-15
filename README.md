# Memory Search MCP

> By [MEOK AI Labs](https://meok.ai) — Persistent AI memory system with semantic search, care-weighted episodes, and knowledge base

## Installation

```bash
pip install memory-search-mcp
```

## Usage

```bash
python server.py
```

## Tools

### `record_memory`
Record a memory episode with care-weighting and emotional valence. Memory types: interaction, insight, decision, emotion.

**Parameters:**
- `content` (str): Memory content
- `source_agent` (str): Source agent (default: "user")
- `memory_type` (str): Type: interaction, insight, decision, emotion (default: "interaction")
- `care_weight` (float): Retrieval priority 0-1 (default: 0.5)
- `importance` (float): Importance 0-1 (default: 0.5)
- `emotional_valence` (float): Emotional valence 0-1 (default: 0.5)
- `tags` (list[str]): Tags for categorization
- `parent_id` (str): Parent episode ID for chains

### `search_memory`
Semantic search across all memories using full-text search with relevance ranking.

**Parameters:**
- `query` (str): Search query
- `limit` (int): Max results (default: 10)
- `care_weight_min` (float): Minimum care weight filter (default: 0.0)
- `memory_type` (str): Filter by type
- `tags` (list[str]): Filter by tags

### `add_knowledge`
Add a knowledge entry to the persistent knowledge base (facts, definitions, reference material).

**Parameters:**
- `topic` (str): Knowledge topic
- `content` (str): Knowledge content
- `source` (str): Source (default: "manual")
- `confidence` (float): Confidence level (default: 0.8)
- `tags` (list[str]): Tags

### `search_knowledge`
Search the knowledge base by topic or content.

**Parameters:**
- `query` (str): Search query
- `limit` (int): Max results (default: 10)
- `min_confidence` (float): Minimum confidence (default: 0.0)

### `list_memories`
List recent memories, optionally filtered by type. Sort by created_at, care_weight, importance, or access_count.

### `get_memory_stats`
Get statistics about the memory store: total count, type breakdown, average care weight, most accessed, storage size.

### `get_temporal_chain`
Follow the temporal chain from a memory episode forward or backward in time.

### `consolidate_memories`
Consolidate old, low-access memories by summarizing and archiving them.

### `semantic_search`
Semantic search using TF-IDF cosine similarity (no external dependencies).

## Authentication

Free tier: 100 calls/day. Upgrade at [meok.ai/pricing](https://meok.ai/pricing) for unlimited access.

## License

MIT — MEOK AI Labs
