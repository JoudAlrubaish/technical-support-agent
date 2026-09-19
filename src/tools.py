import ast
import os
import re
import sqlite3

from pathlib import Path

from langchain_core.tools import tool

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)

from sklearn.metrics.pairwise import (
    cosine_similarity,
)


# =========================================================
# Paths
# =========================================================

DB_PATH = "/tmp/mock_support.db"
UPLOADS_PATH = "/tmp/uploads"

os.makedirs(
    UPLOADS_PATH,
    exist_ok=True,
)


# =========================================================
# Knowledge Base
# =========================================================

KB = [

    {
        "id": "kb-deployment-01",
        "title": "Deployment Guide",
        "text": (
            "The technical support API runs on port 8000 "
            "by default. Expose port 8000 when deploying "
            "the FastAPI service."
        ),
    },

    {
        "id": "kb-network-01",
        "title": "Wi-Fi Troubleshooting",
        "text": (
            "For unstable Wi-Fi, check signal strength, "
            "reconnect to the network, restart the router, "
            "and verify the network adapter."
        ),
    },

    {
        "id": "kb-api-01",
        "title": "API 503 Troubleshooting",
        "text": (
            "HTTP 503 indicates that the service is "
            "temporarily unavailable. Check API health, "
            "database connectivity, and deployment status."
        ),
    },

    {
        "id": "kb-database-01",
        "title": "Database Connection Pool",
        "text": (
            "A database connection pool above 90 percent "
            "utilization may cause request delays or "
            "timeouts. Inspect active connections and "
            "application health."
        ),
    },

    {
        "id": "kb-qlora-01",
        "title": "QLoRA",
        "text": (
            "QLoRA is a parameter-efficient fine-tuning "
            "method for large language models. It keeps "
            "the pretrained model quantized to 4-bit and "
            "trains LoRA adapters, reducing memory "
            "requirements while preserving strong model "
            "performance."
        ),
    },
]


# =========================================================
# Local Documentation
# =========================================================

DOCS = [

    {
        "id": "doc-api",
        "title": "API Documentation",
        "text": (
            "The support API exposes OpenAI-compatible "
            "endpoints on port 8000."
        ),
    },

    {
        "id": "doc-auth",
        "title": "Authentication Documentation",
        "text": (
            "Authentication tokens are used to verify "
            "access to protected APIs."
        ),
    },

    {
        "id": "doc-deployment",
        "title": "Deployment Documentation",
        "text": (
            "The application is deployed using Docker "
            "and exposes port 8000."
        ),
    },
]


# =========================================================
# Package Registry
# =========================================================

PACKAGE_REGISTRY = {

    "torch": {
        "latest_tested": "2.4+",
        "purpose":
            "Deep learning runtime",
    },

    "transformers": {
        "latest_tested": "4.50+",
        "purpose":
            "Hugging Face model loading and inference",
    },

    "langgraph": {
        "latest_tested": "0.2+",
        "purpose":
            "Agent graph orchestration",
    },

    "langchain-core": {
        "latest_tested": "0.3+",
        "purpose":
            "Tool definitions and LangChain interfaces",
    },
}


# =========================================================
# Database Initialization
# =========================================================

def _init_db():

    con = sqlite3.connect(
        DB_PATH
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            priority TEXT DEFAULT 'medium',
            status TEXT DEFAULT 'open',
            resolution TEXT DEFAULT ''
        )
        """
    )

    count = con.execute(
        "SELECT COUNT(*) FROM tickets"
    ).fetchone()[0]

    if count == 0:

        con.executemany(
            """
            INSERT INTO tickets
            (
                title,
                description,
                priority,
                status,
                resolution
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            [

                (
                    "Wi-Fi disconnecting",
                    "User reports repeated Wi-Fi disconnections.",
                    "medium",
                    "closed",
                    "Restarted router and updated network adapter.",
                ),

                (
                    "API returning 503",
                    "Production API became unavailable.",
                    "high",
                    "closed",
                    "Database dependency was degraded.",
                ),
            ],
        )

    con.commit()
    con.close()


_init_db()


# =========================================================
# 1. Calculator
# =========================================================

def _safe_eval(node):

    if isinstance(
        node,
        ast.Expression,
    ):
        return _safe_eval(
            node.body
        )

    if (
        isinstance(
            node,
            ast.Constant,
        )
        and isinstance(
            node.value,
            (int, float),
        )
    ):
        return node.value

    if isinstance(
        node,
        ast.UnaryOp,
    ):

        value = _safe_eval(
            node.operand
        )

        if isinstance(
            node.op,
            ast.USub,
        ):
            return -value

        if isinstance(
            node.op,
            ast.UAdd,
        ):
            return value

    if isinstance(
        node,
        ast.BinOp,
    ):

        left = _safe_eval(
            node.left
        )

        right = _safe_eval(
            node.right
        )

        if isinstance(
            node.op,
            ast.Add,
        ):
            return left + right

        if isinstance(
            node.op,
            ast.Sub,
        ):
            return left - right

        if isinstance(
            node.op,
            ast.Mult,
        ):
            return left * right

        if isinstance(
            node.op,
            ast.Div,
        ):
            return left / right

        if isinstance(
            node.op,
            ast.Mod,
        ):
            return left % right

        if isinstance(
            node.op,
            ast.Pow,
        ):
            return left ** right

    raise ValueError(
        "Unsupported expression"
    )


@tool
def calculator(
    expression: str,
) -> dict:
    """Safely evaluate a restricted arithmetic expression."""

    try:

        tree = ast.parse(
            expression,
            mode="eval",
        )

        result = _safe_eval(
            tree
        )

        return {
            "ok": True,
            "expression": expression,
            "result": result,
        }

    except Exception as e:

        return {
            "ok": False,
            "error": str(e),
        }


# =========================================================
# 2. Log Analyzer
# =========================================================

@tool
def log_analyzer(
    log_text: str,
) -> dict:
    """Extract ERROR, FATAL, EXCEPTION, and WARNING patterns."""

    lines = log_text.splitlines()

    errors = [
        line
        for line in lines
        if re.search(
            r"\b(ERROR|FATAL|EXCEPTION)\b",
            line,
            re.I,
        )
    ]

    warnings = [
        line
        for line in lines
        if re.search(
            r"\bWARN(ING)?\b",
            line,
            re.I,
        )
    ]

    return {
        "errors": errors[:20],
        "warnings": warnings[:20],
        "error_count": len(errors),
        "warning_count": len(warnings),
    }


# =========================================================
# 3. Ticket Search
# =========================================================

@tool
def ticket_search(
    query: str,
) -> dict:
    """Search previous support tickets."""

    con = sqlite3.connect(
        DB_PATH
    )

    rows = con.execute(
        """
        SELECT
            id,
            title,
            status,
            resolution
        FROM tickets
        WHERE title LIKE ?
           OR description LIKE ?
        LIMIT 5
        """,
        (
            f"%{query}%",
            f"%{query}%",
        ),
    ).fetchall()

    con.close()

    tickets = [

        {
            "id": row[0],
            "title": row[1],
            "status": row[2],
            "resolution": row[3],
        }

        for row in rows
    ]

    return {
        "query": query,
        "tickets": tickets,
        "count": len(tickets),
    }


# =========================================================
# 4. Ticket Create
# =========================================================

@tool
def ticket_create(
    title: str,
    description: str,
    priority: str = "medium",
) -> dict:
    """Create a new support ticket."""

    allowed_priorities = {
        "low",
        "medium",
        "high",
        "critical",
    }

    if priority not in allowed_priorities:
        priority = "medium"

    con = sqlite3.connect(
        DB_PATH
    )

    cursor = con.execute(
        """
        INSERT INTO tickets
        (
            title,
            description,
            priority,
            status
        )
        VALUES (?, ?, ?, 'open')
        """,
        (
            title,
            description,
            priority,
        ),
    )

    con.commit()

    ticket_id = (
        cursor.lastrowid
    )

    con.close()

    return {
        "ticket_id": ticket_id,
        "status": "open",
        "priority": priority,
    }


# =========================================================
# 5. System Health Check
# =========================================================

@tool
def system_health_check(
    service: str,
) -> dict:
    """Return deterministic service health for the lab environment."""

    registry = {

        "api": {
            "status": "healthy",
            "latency_ms": 42,
        },

        "database": {
            "status": "degraded",
            "connections_pct": 91,
        },

        "gpu-worker": {
            "status": "healthy",
            "gpu_utilization": 74,
        },

        "network": {
            "status": "healthy",
            "packet_loss_pct": 0.5,
        },
    }

    result = registry.get(
        service.lower(),
        {
            "status": "unknown"
        },
    )

    return {
        "service": service,
        **result,
    }


# =========================================================
# 6. Knowledge Base Search
# =========================================================

@tool
def knowledge_base_search(
    query: str,
) -> dict:
    """Return top trusted KB passages using TF-IDF similarity."""

    documents = [
        item["text"]
        for item in KB
    ]

    vectorizer = (
        TfidfVectorizer(
            stop_words="english"
        )
    )

    matrix = vectorizer.fit_transform(
        documents + [query]
    )

    similarities = cosine_similarity(
        matrix[-1],
        matrix[:-1],
    )[0]

    ranked = (
        similarities
        .argsort()[::-1][:3]
    )

    passages = []

    for idx in ranked:

        item = KB[idx]

        passages.append(
            {
                "source_id":
                    item["id"],

                "title":
                    item["title"],

                "text":
                    item["text"],

                "score":
                    float(
                        similarities[idx]
                    ),
            }
        )

    return {
        "query": query,
        "passages": passages,
    }


# =========================================================
# 7. Documentation Search
# =========================================================

@tool
def documentation_search(
    query: str,
) -> dict:
    """Search local product and API documentation."""

    query_words = set(
        re.findall(
            r"\w+",
            query.lower(),
        )
    )

    results = []

    for doc in DOCS:

        doc_words = set(
            re.findall(
                r"\w+",
                doc["text"].lower(),
            )
        )

        score = len(
            query_words
            & doc_words
        )

        if score > 0:

            results.append(
                {
                    **doc,
                    "score": score,
                }
            )

    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return {
        "query": query,
        "results": results[:5],
    }


# =========================================================
# 8. Package Lookup
# =========================================================

@tool
def package_lookup(
    package_name: str,
    version: str = "",
) -> dict:
    """Look up package and version information."""

    package = (
        PACKAGE_REGISTRY.get(
            package_name.lower()
        )
    )

    if not package:

        return {
            "found": False,
            "package": package_name,
        }

    return {
        "found": True,
        "package": package_name,
        "requested_version": version,
        **package,
    }


# =========================================================
# 9. Read-Only SQL
# =========================================================

@tool
def sql_query(
    query: str,
) -> dict:
    """Execute an allowlisted read-only SQL query."""

    normalized = (
        query
        .strip()
        .lower()
    )

    if not normalized.startswith(
        "select"
    ):

        return {
            "ok": False,
            "error":
                "Only SELECT queries are allowed.",
        }

    if ";" in normalized[:-1]:

        return {
            "ok": False,
            "error":
                "Multiple SQL statements are not allowed.",
        }

    if "tickets" not in normalized:

        return {
            "ok": False,
            "error":
                "Only the tickets table is allowlisted.",
        }

    try:

        con = sqlite3.connect(
            DB_PATH
        )

        cursor = con.execute(
            query
        )

        columns = [
            col[0]
            for col
            in cursor.description
        ]

        rows = cursor.fetchmany(
            20
        )

        con.close()

        results = [
            dict(
                zip(
                    columns,
                    row,
                )
            )
            for row in rows
        ]

        return {
            "ok": True,
            "rows": results,
            "count": len(results),
        }

    except Exception as e:

        return {
            "ok": False,
            "error": str(e),
        }


# =========================================================
# 10. File Search
# =========================================================

@tool
def file_search(
    query: str,
    path: str = UPLOADS_PATH,
) -> dict:
    """Search approved uploaded text/log/config files."""

    approved_root = Path(
        UPLOADS_PATH
    ).resolve()

    requested_path = Path(
        path
    ).resolve()

    if (
        approved_root
        not in requested_path.parents
        and requested_path
        != approved_root
    ):

        return {
            "ok": False,
            "error":
                "Path is outside the approved uploads directory.",
        }

    matches = []

    if not requested_path.exists():

        return {
            "ok": True,
            "matches": [],
        }

    for file in requested_path.rglob(
        "*"
    ):

        if not file.is_file():
            continue

        if file.suffix.lower() not in {
            ".txt",
            ".log",
            ".json",
            ".yaml",
            ".yml",
            ".conf",
        }:
            continue

        try:

            content = file.read_text(
                encoding="utf-8",
                errors="ignore",
            )

            if (
                query.lower()
                in content.lower()
            ):

                matches.append(
                    {
                        "file":
                            str(file),

                        "snippet":
                            content[:500],
                    }
                )

        except Exception:
            continue

    return {
        "ok": True,
        "query": query,
        "matches": matches[:10],
    }


# =========================================================
# 11. Web Search — Mock
# =========================================================

@tool
def web_search(
    query: str,
) -> dict:
    """External technical fallback using a deterministic lab mock."""

    return {
        "query": query,
        "provider": "mock",

        "results": [
            {
                "title":
                    "Technical Support Reference",

                "snippet":
                    "External search is mocked "
                    "in the local lab environment.",

                "url":
                    None,
            }
        ],
    }


# =========================================================
# 12. Human Escalation
# =========================================================

@tool
def escalate_to_human(
    reason: str,
    evidence: str,
) -> dict:
    """Create a high-priority escalation ticket."""

    ticket = ticket_create.invoke(
        {
            "title":
                f"ESCALATION: {reason}",

            "description":
                evidence,

            "priority":
                "high",
        }
    )

    return {
        "escalated": True,
        "reason": reason,
        "ticket": ticket,
    }


# =========================================================
# 13. Diagnostic Runbook
# =========================================================

@tool
def diagnostic_runbook(
    issue_type: str,
    service: str,
    symptom: str,
) -> dict:
    """Run deterministic diagnostic steps before model reasoning."""

    steps = []

    health = (
        system_health_check.invoke(
            {
                "service": service,
            }
        )
    )

    steps.append(
        {
            "step":
                "health_check",

            "result":
                health,
        }
    )

    previous_tickets = (
        ticket_search.invoke(
            {
                "query":
                    issue_type,
            }
        )
    )

    steps.append(
        {
            "step":
                "ticket_search",

            "result":
                previous_tickets,
        }
    )

    kb = (
        knowledge_base_search.invoke(
            {
                "query":
                    symptom,
            }
        )
    )

    steps.append(
        {
            "step":
                "knowledge_base_search",

            "result":
                kb,
        }
    )

    requires_escalation = (

        health.get(
            "status"
        )
        == "down"

        or any(
            phrase
            in symptom.lower()

            for phrase in [
                "data loss",
                "security breach",
                "corruption",
                "production down",
            ]
        )
    )

    return {
        "issue_type":
            issue_type,

        "service":
            service,

        "symptom":
            symptom,

        "steps":
            steps,

        "requires_escalation":
            requires_escalation,
    }


# =========================================================
# Registry
# =========================================================

ALL_TOOLS = [
    knowledge_base_search,
    ticket_search,
    ticket_create,
    system_health_check,
    log_analyzer,
    documentation_search,
    package_lookup,
    sql_query,
    calculator,
    file_search,
    web_search,
    escalate_to_human,
    diagnostic_runbook,
]