# Memory Search MCP Server

Persistent memory system for AI agents and assistants. Record episodic memories with care-weighting and emotional valence, search with full-text relevance ranking, maintain a knowledge base of facts and reference material, follow temporal chains, and consolidate old memories automatically.

Zero external dependencies beyond the `mcp` package -- uses SQLite with FTS5 for fast full-text search. Data persists in `~/.mcp-memory/memories.db`.

## Tools

| Tool | Description |
|------|-------------|
| `record_memory` | Store a memory episode with care weight, importance, emotion, and tags |
| `search_memory` | Full-text semantic search with care weight and tag filtering |
| `add_knowledge` | Add persistent facts/reference material to the knowledge base |
| `search_knowledge` | Search the knowledge base by topic or content |
| `list_memories` | Browse recent memories sorted by time, importance, or access count |
| `get_memory_stats` | Memory store statistics: counts, averages, storage size |
| `get_temporal_chain` | Follow the timeline forward/backward from any memory |
| `consolidate_memories` | Archive old low-access memories to save space |

## Installation

```bash
pip install mcp
```

## Usage

### Run the server

```bash
python server.py
```

### Claude Desktop config

```json
{
  "mcpServers": {
    "memory-search": {
      "command": "python",
      "args": ["/path/to/memory-search-mcp/server.py"]
    }
  }
}
```

### Example calls

**Record a memory:**
```
Tool: record_memory
Input: {"content": "User prefers dark mode and compact layouts", "source_agent": "preferences", "memory_type": "insight", "care_weight": 0.8, "tags": ["preferences", "ui"]}
Output: {"success": true, "episode_id": "a3f2b1c8d9e0", "timestamp": "2026-04-13T10:30:00"}
```

**Search memories:**
```
Tool: search_memory
Input: {"query": "user interface preferences", "limit": 5, "care_weight_min": 0.5}
Output: {"results": [...], "count": 3, "query": "user interface preferences"}
```

**Add knowledge:**
```
Tool: add_knowledge
Input: {"topic": "Python asyncio", "content": "Use asyncio.gather() for concurrent coroutines...", "confidence": 0.9}
Output: {"success": true, "knowledge_id": "k1a2b3c4d5e6", "topic": "Python asyncio"}
```

**Follow temporal chain:**
```
Tool: get_temporal_chain
Input: {"episode_id": "a3f2b1c8d9e0", "direction": "backward", "max_steps": 10}
Output: {"chain": [...], "direction": "backward", "steps": 7}
```

## Data Storage

All data is stored in `~/.mcp-memory/memories.db` (SQLite). To reset, simply delete this file. To back up, copy it.

## Pricing

| Tier | Limit | Price |
|------|-------|-------|
| Free | 100 calls/day | $0 |
| Pro | Unlimited + vector embeddings + cloud sync | $9/mo |
| Enterprise | Custom + team sharing + encryption at rest | Contact us |

## License

MIT
