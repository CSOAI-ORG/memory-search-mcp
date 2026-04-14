#!/usr/bin/env python3
"""
Memory Search MCP Server
=========================
Persistent memory system for AI agents with semantic search, care-weighted
episodes, temporal chains, tagging, and memory consolidation. Uses SQLite
with FTS5 for full-text search (zero external dependencies beyond stdlib).

Install: pip install mcp
Run:     python server.py
"""

import json
import math
import sqlite3
import uuid
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from collections import defaultdict
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
FREE_DAILY_LIMIT = 100
_usage: dict[str, list[datetime]] = defaultdict(list)


def _check_rate_limit(caller: str = "anonymous") -> Optional[str]:
    now = datetime.now()
    cutoff = now - timedelta(days=1)
    _usage[caller] = [t for t in _usage[caller] if t > cutoff]
    if len(_usage[caller]) >= FREE_DAILY_LIMIT:
        return f"Free tier limit reached ({FREE_DAILY_LIMIT}/day). Upgrade to Pro: https://mcpize.com/memory-search-mcp/pro"
    _usage[caller].append(now)
    return None


# ---------------------------------------------------------------------------
# Memory Store (SQLite + FTS5)
# ---------------------------------------------------------------------------
DATA_DIR = Path.home() / ".mcp-memory"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "memories.db"


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    # Create tables on first use
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            source_agent TEXT DEFAULT 'unknown',
            memory_type TEXT DEFAULT 'interaction',
            care_weight REAL DEFAULT 0.5,
            emotional_valence REAL DEFAULT 0.5,
            importance REAL DEFAULT 0.5,
            tags TEXT DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            access_count INTEGER DEFAULT 0,
            last_accessed TEXT,
            parent_id TEXT,
            FOREIGN KEY (parent_id) REFERENCES memories(id)
        );

        CREATE TABLE IF NOT EXISTS knowledge (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT DEFAULT 'manual',
            confidence REAL DEFAULT 0.8,
            tags TEXT DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            access_count INTEGER DEFAULT 0
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content, source_agent, memory_type, tags,
            content='memories',
            content_rowid='rowid'
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
            topic, content, source, tags,
            content='knowledge',
            content_rowid='rowid'
        );

        -- Triggers to keep FTS in sync
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, source_agent, memory_type, tags)
            VALUES (new.rowid, new.content, new.source_agent, new.memory_type, new.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, source_agent, memory_type, tags)
            VALUES ('delete', old.rowid, old.content, old.source_agent, old.memory_type, old.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, source_agent, memory_type, tags)
            VALUES ('delete', old.rowid, old.content, old.source_agent, old.memory_type, old.tags);
            INSERT INTO memories_fts(rowid, content, source_agent, memory_type, tags)
            VALUES (new.rowid, new.content, new.source_agent, new.memory_type, new.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge BEGIN
            INSERT INTO knowledge_fts(rowid, topic, content, source, tags)
            VALUES (new.rowid, new.topic, new.content, new.source, new.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge BEGIN
            INSERT INTO knowledge_fts(knowledge_fts, rowid, topic, content, source, tags)
            VALUES ('delete', old.rowid, old.topic, old.content, old.source, old.tags);
        END;
    """)
    return conn


def _simple_relevance(query: str, text: str) -> float:
    """Simple TF-based relevance scoring for ranking results."""
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    text_words = re.findall(r'\b\w+\b', text.lower())
    if not query_words or not text_words:
        return 0.0
    matches = sum(1 for w in text_words if w in query_words)
    return matches / len(text_words)


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Memory Search MCP",
    instructions="Persistent AI memory system with semantic search, care-weighted episodes, temporal chains, knowledge base, and memory consolidation.")


@mcp.tool()
def record_memory(
    content: str,
    source_agent: str = "user",
    memory_type: str = "interaction",
    care_weight: float = 0.5,
    importance: float = 0.5,
    emotional_valence: float = 0.5,
    tags: list[str] | None = None,
    parent_id: str | None = None) -> dict:
    """Record a memory episode with care-weighting and emotional valence.
    Memory types: interaction, insight, decision, emotion.
    Care weight (0-1) determines retrieval priority.
    Importance below 0.2 with care_weight below 0.3 will be rejected (noise filter)."""
    err = _check_rate_limit()
    if err:
        return {"error": err}

    if importance < 0.2 and care_weight < 0.3:
        return {"success": False, "reason": "Below minimum importance threshold (0.2)"}

    mid = str(uuid.uuid4())[:12]
    now = datetime.now().isoformat()
    tag_json = json.dumps(tags or [])

    db = _get_db()
    db.execute(
        """INSERT INTO memories (id, content, source_agent, memory_type, care_weight,
           emotional_valence, importance, tags, created_at, updated_at, parent_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (mid, content, source_agent, memory_type, care_weight,
         emotional_valence, importance, tag_json, now, now, parent_id))
    db.commit()
    db.close()
    return {"success": True, "episode_id": mid, "timestamp": now}


@mcp.tool()
def search_memory(query: str, limit: int = 10, care_weight_min: float = 0.0,
                   memory_type: str | None = None, tags: list[str] | None = None) -> dict:
    """Semantic search across all memories using full-text search with relevance ranking.
    Filter by minimum care weight, memory type, and tags."""
    err = _check_rate_limit()
    if err:
        return {"error": err}

    db = _get_db()

    # FTS5 search
    try:
        # Escape FTS5 special chars
        fts_query = re.sub(r'[^\w\s]', '', query)
        fts_terms = ' OR '.join(fts_query.split())
        rows = db.execute(
            """SELECT m.* FROM memories m
               JOIN memories_fts f ON m.rowid = f.rowid
               WHERE memories_fts MATCH ?
               AND m.care_weight >= ?
               ORDER BY rank
               LIMIT ?""",
            (fts_terms, care_weight_min, limit * 3),  # Over-fetch for filtering
        ).fetchall()
    except Exception:
        # Fallback to LIKE search
        rows = db.execute(
            """SELECT * FROM memories WHERE content LIKE ? AND care_weight >= ?
               ORDER BY created_at DESC LIMIT ?""",
            (f"%{query}%", care_weight_min, limit * 3)).fetchall()

    results = []
    for row in rows:
        row_tags = json.loads(row["tags"]) if row["tags"] else []
        # Filter by type
        if memory_type and row["memory_type"] != memory_type:
            continue
        # Filter by tags
        if tags and not any(t in row_tags for t in tags):
            continue
        results.append({
            "id": row["id"],
            "content": row["content"],
            "source_agent": row["source_agent"],
            "memory_type": row["memory_type"],
            "care_weight": row["care_weight"],
            "emotional_valence": row["emotional_valence"],
            "importance": row["importance"],
            "tags": row_tags,
            "created_at": row["created_at"],
        })
        if len(results) >= limit:
            break

    # Update access counts
    for r in results:
        db.execute(
            "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (datetime.now().isoformat(), r["id"]))
    db.commit()
    db.close()
    return {"results": results, "count": len(results), "query": query}


@mcp.tool()
def add_knowledge(topic: str, content: str, source: str = "manual",
                   confidence: float = 0.8, tags: list[str] | None = None) -> dict:
    """Add a knowledge entry to the persistent knowledge base.
    Knowledge is separate from episodic memory -- it stores facts, definitions,
    and reference material that persists across sessions."""
    err = _check_rate_limit()
    if err:
        return {"error": err}

    kid = str(uuid.uuid4())[:12]
    now = datetime.now().isoformat()
    tag_json = json.dumps(tags or [])

    db = _get_db()
    db.execute(
        """INSERT INTO knowledge (id, topic, content, source, confidence, tags, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (kid, topic, content, source, confidence, tag_json, now, now))
    db.commit()
    db.close()
    return {"success": True, "knowledge_id": kid, "topic": topic}


@mcp.tool()
def search_knowledge(query: str, limit: int = 10, min_confidence: float = 0.0) -> dict:
    """Search the knowledge base by topic or content. Returns facts and reference
    material ranked by relevance and confidence."""
    err = _check_rate_limit()
    if err:
        return {"error": err}

    db = _get_db()
    try:
        fts_query = re.sub(r'[^\w\s]', '', query)
        fts_terms = ' OR '.join(fts_query.split())
        rows = db.execute(
            """SELECT k.* FROM knowledge k
               JOIN knowledge_fts f ON k.rowid = f.rowid
               WHERE knowledge_fts MATCH ?
               AND k.confidence >= ?
               ORDER BY rank
               LIMIT ?""",
            (fts_terms, min_confidence, limit)).fetchall()
    except Exception:
        rows = db.execute(
            """SELECT * FROM knowledge WHERE (topic LIKE ? OR content LIKE ?)
               AND confidence >= ? ORDER BY created_at DESC LIMIT ?""",
            (f"%{query}%", f"%{query}%", min_confidence, limit)).fetchall()

    results = []
    for row in rows:
        results.append({
            "id": row["id"],
            "topic": row["topic"],
            "content": row["content"],
            "source": row["source"],
            "confidence": row["confidence"],
            "tags": json.loads(row["tags"]) if row["tags"] else [],
            "created_at": row["created_at"],
        })
        db.execute(
            "UPDATE knowledge SET access_count = access_count + 1 WHERE id = ?",
            (row["id"]))

    db.commit()
    db.close()
    return {"results": results, "count": len(results)}


@mcp.tool()
def list_memories(limit: int = 50, memory_type: str | None = None,
                   sort_by: str = "created_at") -> dict:
    """List recent memories, optionally filtered by type. Sort by created_at,
    care_weight, importance, or access_count."""
    err = _check_rate_limit()
    if err:
        return {"error": err}

    valid_sorts = {"created_at", "care_weight", "importance", "access_count"}
    if sort_by not in valid_sorts:
        sort_by = "created_at"

    db = _get_db()
    if memory_type:
        rows = db.execute(
            f"SELECT * FROM memories WHERE memory_type = ? ORDER BY {sort_by} DESC LIMIT ?",
            (memory_type, limit)).fetchall()
    else:
        rows = db.execute(
            f"SELECT * FROM memories ORDER BY {sort_by} DESC LIMIT ?", (limit)).fetchall()

    memories = []
    for row in rows:
        memories.append({
            "id": row["id"],
            "content": row["content"][:200],
            "source_agent": row["source_agent"],
            "memory_type": row["memory_type"],
            "care_weight": row["care_weight"],
            "importance": row["importance"],
            "tags": json.loads(row["tags"]) if row["tags"] else [],
            "created_at": row["created_at"],
            "access_count": row["access_count"],
        })
    db.close()
    return {"memories": memories, "count": len(memories)}


@mcp.tool()
def get_memory_stats() -> dict:
    """Get statistics about the memory store: total count, type breakdown,
    average care weight, most accessed, and storage size."""
    db = _get_db()

    total = db.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
    knowledge_total = db.execute("SELECT COUNT(*) as c FROM knowledge").fetchone()["c"]

    type_breakdown = {}
    for row in db.execute("SELECT memory_type, COUNT(*) as c FROM memories GROUP BY memory_type"):
        type_breakdown[row["memory_type"]] = row["c"]

    avg_care = db.execute("SELECT AVG(care_weight) as a FROM memories").fetchone()["a"] or 0
    avg_importance = db.execute("SELECT AVG(importance) as a FROM memories").fetchone()["a"] or 0

    most_accessed = db.execute(
        "SELECT id, content, access_count FROM memories ORDER BY access_count DESC LIMIT 5"
    ).fetchall()

    db.close()

    db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0

    return {
        "total_memories": total,
        "total_knowledge": knowledge_total,
        "type_breakdown": type_breakdown,
        "avg_care_weight": round(avg_care, 3),
        "avg_importance": round(avg_importance, 3),
        "most_accessed": [
            {"id": r["id"], "content": r["content"][:100], "access_count": r["access_count"]}
            for r in most_accessed
        ],
        "storage_bytes": db_size,
        "storage_mb": round(db_size / 1024 / 1024, 2),
        "db_path": str(DB_PATH),
    }


@mcp.tool()
def get_temporal_chain(episode_id: str, direction: str = "forward", max_steps: int = 5) -> dict:
    """Follow the temporal chain from a memory episode forward or backward in time.
    Useful for understanding the context and sequence of events."""
    err = _check_rate_limit()
    if err:
        return {"error": err}

    db = _get_db()
    # Get the anchor memory
    anchor = db.execute("SELECT * FROM memories WHERE id = ?", (episode_id)).fetchone()
    if not anchor:
        db.close()
        return {"error": f"Episode {episode_id} not found"}

    anchor_time = anchor["created_at"]
    op = ">" if direction == "forward" else "<"
    order = "ASC" if direction == "forward" else "DESC"

    rows = db.execute(
        f"SELECT * FROM memories WHERE created_at {op} ? ORDER BY created_at {order} LIMIT ?",
        (anchor_time, max_steps)).fetchall()

    chain = [{
        "id": anchor["id"],
        "content": anchor["content"][:200],
        "created_at": anchor["created_at"],
        "memory_type": anchor["memory_type"],
        "position": "anchor",
    }]
    for i, row in enumerate(rows):
        chain.append({
            "id": row["id"],
            "content": row["content"][:200],
            "created_at": row["created_at"],
            "memory_type": row["memory_type"],
            "position": i + 1,
        })

    db.close()
    return {"chain": chain, "direction": direction, "steps": len(chain) - 1}


@mcp.tool()
def consolidate_memories(older_than_days: int = 30, min_access: int = 0) -> dict:
    """Consolidate old, low-access memories by summarizing and archiving them.
    Memories older than the threshold with access_count <= min_access get archived."""
    err = _check_rate_limit()
    if err:
        return {"error": err}

    cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()
    db = _get_db()

    candidates = db.execute(
        "SELECT * FROM memories WHERE created_at < ? AND access_count <= ? ORDER BY created_at",
        (cutoff, min_access)).fetchall()

    if not candidates:
        db.close()
        return {"consolidated": 0, "message": "No memories eligible for consolidation"}

    # Archive to a consolidated entry
    contents = [f"[{r['memory_type']}] {r['content'][:100]}" for r in candidates]
    summary = f"Consolidated {len(candidates)} memories from before {cutoff[:10]}:\n" + "\n".join(contents[:20])
    if len(contents) > 20:
        summary += f"\n... and {len(contents) - 20} more"

    cid = str(uuid.uuid4())[:12]
    now = datetime.now().isoformat()
    db.execute(
        """INSERT INTO memories (id, content, source_agent, memory_type, care_weight,
           importance, tags, created_at, updated_at)
           VALUES (?, ?, 'system', 'insight', 0.6, 0.7, '["consolidated"]', ?, ?)""",
        (cid, summary, now, now))

    # Delete originals
    ids = [r["id"] for r in candidates]
    db.execute(f"DELETE FROM memories WHERE id IN ({','.join('?' * len(ids))})", ids)
    db.commit()
    db.close()

    return {
        "consolidated": len(candidates),
        "archive_id": cid,
        "summary_preview": summary[:300],
    }


if __name__ == "__main__":
    mcp.run()
