import uuid

from typing import (
    TypedDict,
    Literal,
)

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from src.routers import (
    hybrid_router,
)

from src.tools import (
    knowledge_base_search,
    diagnostic_runbook,
    escalate_to_human,
)

from src.specialists import (
    run_qa_model,
    run_support_model,
)


# =========================================================
# Agent State
# =========================================================

class SupportState(
    TypedDict,
    total=False,
):

    user_message: str

    route: str
    router_source: str

    intent: str
    confidence: float

    context: list
    tool_results: list

    answer: str

    escalated: bool

    trace_id: str

    router_latency_ms: float


# =========================================================
# Helpers
# =========================================================

def infer_service(
    text: str,
) -> str:

    text = text.lower()

    if any(
        word in text
        for word in [
            "database",
            "postgres",
            "sql",
            "connection pool",
        ]
    ):
        return "database"

    if any(
        word in text
        for word in [
            "gpu",
            "cuda",
        ]
    ):
        return "gpu-worker"

    if any(
        word in text
        for word in [
            "wifi",
            "wi-fi",
            "network",
            "internet",
        ]
    ):
        return "network"

    return "api"


# =========================================================
# Router Node
# =========================================================

def route_node(
    state: SupportState,
):

    decision = hybrid_router(
        state["user_message"]
    )

    return {

        "route":
            decision["route"],

        "router_source":
            decision.get(
                "source",
                "",
            ),

        "intent":
            decision.get(
                "intent",
                "",
            ),

        "confidence":
            decision.get(
                "confidence",
                decision.get(
                    "classifier_confidence",
                    0.0,
                ),
            ),

        "router_latency_ms":
            decision.get(
                "latency_ms",
                0.0,
            ),

        "trace_id":
            state.get(
                "trace_id",
                str(
                    uuid.uuid4()
                ),
            ),
    }


# =========================================================
# QA Node — Model B
# =========================================================

def qa_node(
    state: SupportState,
):

    kb_result = (
        knowledge_base_search.invoke(
            {
                "query":
                    state[
                        "user_message"
                    ],
            }
        )
    )

    passages = (
        kb_result.get(
            "passages",
            [],
        )
    )

    # Use only highest-ranked trusted passage
    top_passages = (
        passages[:1]
    )

    qa_result = (
        run_qa_model(
            question=
                state[
                    "user_message"
                ],

            contexts=
                top_passages,
        )
    )

    return {

        "context":
            top_passages,

        "answer":
            qa_result[
                "answer"
            ],
    }


# =========================================================
# Tools Node
# =========================================================

def tool_node(
    state: SupportState,
):

    service = infer_service(
        state[
            "user_message"
        ]
    )

    result = (
        diagnostic_runbook.invoke(
            {

                "issue_type":
                    state.get(
                        "intent",
                        "unknown",
                    ),

                "service":
                    service,

                "symptom":
                    state[
                        "user_message"
                    ],
            }
        )
    )

    return {
        "tool_results": [
            result
        ]
    }


# =========================================================
# Model C Support Node
# =========================================================

def support_node(
    state: SupportState,
):

    context = state.get(
        "context",
        [],
    )

    # Retrieve useful KB context for support questions
    if not context:

        kb_result = (
            knowledge_base_search.invoke(
                {
                    "query":
                        state[
                            "user_message"
                        ],
                }
            )
        )

        context = [

            passage

            for passage
            in kb_result.get(
                "passages",
                [],
            )

            if (
                passage.get(
                    "score",
                    0,
                )
                > 0.10
            )

        ][:2]

    answer = run_support_model(

        user_message=
            state[
                "user_message"
            ],

        context=
            context,

        tool_results=
            state.get(
                "tool_results",
                [],
            ),
    )

    return {

        "context":
            context,

        "answer":
            answer,
    }


# =========================================================
# Human Escalation
# =========================================================

def escalation_node(
    state: SupportState,
):

    result = (
        escalate_to_human.invoke(
            {

                "reason":
                    "High-risk or unresolved "
                    "technical incident",

                "evidence":
                    state[
                        "user_message"
                    ],
            }
        )
    )

    ticket = result.get(
        "ticket",
        {},
    )

    ticket_id = ticket.get(
        "ticket_id"
    )

    return {

        "escalated":
            True,

        "answer":
            (
                "This issue has been escalated "
                "to human support. "
                f"Ticket ID: {ticket_id}"
            ),
    }


# =========================================================
# Conditional Routing
# =========================================================

def choose_after_router(
    state: SupportState,
) -> Literal[
    "qa",
    "tools",
    "support",
    "escalate",
]:

    route = state.get(
        "route"
    )

    if route == "qa":
        return "qa"

    if route == "tools":
        return "tools"

    if route == "escalate":
        return "escalate"

    return "support"


# =========================================================
# Build LangGraph
# =========================================================

builder = StateGraph(
    SupportState
)

builder.add_node(
    "route",
    route_node,
)

builder.add_node(
    "qa",
    qa_node,
)

builder.add_node(
    "tools",
    tool_node,
)

builder.add_node(
    "support",
    support_node,
)

builder.add_node(
    "escalate",
    escalation_node,
)


builder.add_edge(
    START,
    "route",
)


builder.add_conditional_edges(
    "route",
    choose_after_router,
    {
        "qa":
            "qa",

        "tools":
            "tools",

        "support":
            "support",

        "escalate":
            "escalate",
    },
)


builder.add_edge(
    "qa",
    END,
)

builder.add_edge(
    "tools",
    "support",
)

builder.add_edge(
    "support",
    END,
)

builder.add_edge(
    "escalate",
    END,
)


graph = builder.compile()


# =========================================================
# Unified Agent
# =========================================================

def run_agent(
    user_message: str,
):

    return graph.invoke(
        {

            "user_message":
                user_message,

            "trace_id":
                str(
                    uuid.uuid4()
                ),
        }
    )