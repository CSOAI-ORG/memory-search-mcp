# Memory Search MCP Server

> **By [MEOK AI Labs](https://meok.ai)** — Sovereign AI tools for everyone.

Persistent memory system for AI agents and assistants. Record episodic memories with care-weighting and emotional valence, search with full-text relevance ranking, maintain a knowledge base, follow temporal chains, and consolidate old memories automatically.

[![MCPize](https://img.shields.io/badge/MCPize-Listed-blue)](https://mcpize.com/mcp/memory-search)
[![MIT License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![MEOK AI Labs](https://img.shields.io/badge/MEOK_AI_Labs-255+_servers-purple)](https://meok.ai)

## Tools

| Tool | Description |
|------|-------------|
| `record_memory` | Record a memory episode with care-weighting and emotional valence |
| `search_memory` | Semantic search across all memories with relevance ranking |
| `add_knowledge` | Add a knowledge entry to the persistent knowledge base |
| `search_knowledge` | Search the knowledge base by topic or content |
| `list_memories` | List recent memories, optionally filtered by type |
| `get_memory_stats` | Get statistics about the memory store |
| `get_temporal_chain` | Follow the temporal chain from a memory forward or backward |
| `consolidate_memories` | Consolidate old, low-access memories by summarizing them |

## Quick Start

```bash
pip install mcp
git clone https://github.com/CSOAI-ORG/memory-search-mcp.git
cd memory-search-mcp
python server.py
```

## Claude Desktop Config

```json
{
  "mcpServers": {
    "memory-search": {
      "command": "python",
      "args": ["server.py"],
      "cwd": "/path/to/memory-search-mcp"
    }
  }
}
```

## Pricing

| Plan | Price | Requests |
|------|-------|----------|
| Free | $0/mo | 100 requests/day |
| Pro | $9/mo | Unlimited + vector embeddings + cloud sync |
| Enterprise | Contact us | Custom + team sharing + encryption at rest |

[Get on MCPize](https://mcpize.com/mcp/memory-search)

## Part of MEOK AI Labs

This is one of 255+ MCP servers by MEOK AI Labs. Browse all at [meok.ai](https://meok.ai) or [GitHub](https://github.com/CSOAI-ORG).

---
**MEOK AI Labs** | [meok.ai](https://meok.ai) | nicholas@meok.ai | United Kingdom
