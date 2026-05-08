<div align="center">

# Memory Search MCP

**MCP server for memory search mcp operations**

[![PyPI](https://img.shields.io/pypi/v/meok-memory-search-mcp)](https://pypi.org/project/meok-memory-search-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MEOK AI Labs](https://img.shields.io/badge/MEOK_AI_Labs-MCP_Server-purple)](https://meok.ai)

</div>

## Overview

Memory Search MCP provides AI-powered tools via the Model Context Protocol (MCP).

## Tools

| Tool | Description |
|------|-------------|
| `record_memory` | Record a memory episode with care-weighting and emotional valence. |
| `search_memory` | Semantic search across all memories using full-text search with relevance rankin |
| `add_knowledge` | Add a knowledge entry to the persistent knowledge base. |
| `search_knowledge` | Search the knowledge base by topic or content. Returns facts and reference |
| `list_memories` | List recent memories, optionally filtered by type. Sort by created_at, |
| `get_memory_stats` | Get statistics about the memory store: total count, type breakdown, |
| `get_temporal_chain` | Follow the temporal chain from a memory episode forward or backward in time. |
| `consolidate_memories` | Consolidate old, low-access memories by summarizing and archiving them. |
| `semantic_search` | Semantic search using TF-IDF cosine similarity (no external deps). |

## Installation

```bash
pip install meok-memory-search-mcp
```

## Usage with Claude Desktop

Add to your Claude Desktop MCP config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "memory-search-mcp": {
      "command": "python",
      "args": ["-m", "meok_memory_search_mcp.server"]
    }
  }
}
```

## Usage with FastMCP

```python
from mcp.server.fastmcp import FastMCP

# This server exposes 9 tool(s) via MCP
# See server.py for full implementation
```

## License

MIT © [MEOK AI Labs](https://meok.ai)
