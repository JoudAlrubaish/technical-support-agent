import uuid

from langfuse import (
    get_client,
    propagate_attributes,
)

from langfuse.langchain import (
    CallbackHandler,
)

from src.graph import graph


langfuse = get_client()


def run_traced_agent(
    user_message: str,
):

    trace_id = str(
        uuid.uuid4()
    )

    handler = CallbackHandler()

    with langfuse.start_as_current_observation(
        name="technical-support-agent",
        as_type="agent",
        input={
            "user_message":
                user_message
        },
    ) as observation:

        with propagate_attributes(
            session_id=trace_id,
            tags=[
                "technical-support",
                "langgraph",
            ],
            metadata={
                "router":
                    "hybrid",
                "application":
                    "technical-support-agent",
            },
        ):

            result = graph.invoke(
                {
                    "user_message":
                        user_message,

                    "trace_id":
                        trace_id,
                },

                config={
                    "callbacks": [
                        handler
                    ],

                    "metadata": {
                        "langfuse_tags": [
                            "technical-support",
                            "langgraph",
                        ]
                    },
                },
            )

        observation.update(
            output={
                "route":
                    result.get(
                        "route"
                    ),

                "router_source":
                    result.get(
                        "router_source"
                    ),

                "answer":
                    result.get(
                        "answer"
                    ),
            }
        )

    # Important for short terminal runs
    langfuse.flush()

    return result