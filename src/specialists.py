import json
import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
)


# =========================================================
# Models
# =========================================================

MODEL_B_REPO = (
    "JoudAlrubaish/"
    "technical-support-extractive-qa"
)

MODEL_C_REPO = (
    "JoudAlrubaish/"
    "technical-support-agent-model-c"
)


# =========================================================
# Model B — Extractive QA
# =========================================================

qa_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_B_REPO
)

qa_model = AutoModelForQuestionAnswering.from_pretrained(
    MODEL_B_REPO
)

qa_model.eval()


def _extract_answer(
    question: str,
    context: str,
) -> dict:

    encoded = qa_tokenizer(
        question,
        context,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation="only_second",
        max_length=512,
    )

    offsets = encoded.pop(
        "offset_mapping"
    )[0]

    sequence_ids = encoded.sequence_ids(0)

    with torch.no_grad():
        outputs = qa_model(
            **encoded
        )

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    context_mask = torch.tensor(
        [
            seq_id == 1
            for seq_id in sequence_ids
        ],
        dtype=torch.bool,
    )

    start_logits = start_logits.masked_fill(
        ~context_mask,
        float("-inf"),
    )

    end_logits = end_logits.masked_fill(
        ~context_mask,
        float("-inf"),
    )

    best_score = float("-inf")
    best_start = None
    best_end = None

    valid_positions = torch.where(
        context_mask
    )[0].tolist()

    for start_idx in valid_positions:

        for end_idx in valid_positions:

            if end_idx < start_idx:
                continue

            if end_idx - start_idx > 30:
                continue

            score = (
                start_logits[start_idx]
                + end_logits[end_idx]
            ).item()

            if score > best_score:

                best_score = score
                best_start = start_idx
                best_end = end_idx

    if best_start is None:

        return {
            "answer": "",
            "score": 0.0,
        }

    start_char = int(
        offsets[best_start][0]
    )

    end_char = int(
        offsets[best_end][1]
    )

    answer = context[
        start_char:end_char
    ].strip()

    return {
        "answer": answer,
        "score": float(best_score),
    }


def run_qa_model(
    question: str,
    contexts: list,
) -> dict:

    if not contexts:

        return {
            "answer": "",
            "score": 0.0,
            "source_id": None,
        }

    best_result = {
        "answer": "",
        "score": float("-inf"),
        "source_id": None,
    }

    for item in contexts:

        if isinstance(item, dict):

            context_text = item.get(
                "text",
                "",
            )

            source_id = item.get(
                "source_id"
            )

        else:

            context_text = str(item)
            source_id = None

        if not context_text:
            continue

        result = _extract_answer(
            question,
            context_text,
        )

        if (
            result["score"]
            > best_result["score"]
        ):

            best_result = {
                "answer":
                    result["answer"],

                "score":
                    result["score"],

                "source_id":
                    source_id,
            }

    return best_result


# =========================================================
# Model C — Support Specialist
# =========================================================

support_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_C_REPO
)

support_model = AutoModelForCausalLM.from_pretrained(
    MODEL_C_REPO
)

support_model.eval()


SUPPORT_SYSTEM = """
You are an IT technical support specialist.

Provide clear, concise, safe, and practical troubleshooting guidance.

Use provided trusted context and tool results when available.

When trusted context is provided, prioritize it over prior knowledge.

Do not invent technical facts.

Avoid repeating the same sentence.

Do not claim certainty when information is insufficient.

Escalate high-risk or unresolved issues to human support.
"""


# =========================================================
# Guardrails
# =========================================================

def _needs_uncertainty_guardrail(
    user_message: str,
) -> bool:

    text = user_message.lower()

    phrases = [
        "exact cause",
        "exactly what caused",
        "exactly caused",
        "not enough information",
        "no other information",
        "cannot provide more information",
    ]

    return any(
        phrase in text
        for phrase in phrases
    )


def _is_factual_question(
    user_message: str,
) -> bool:

    text = user_message.strip().lower()

    prefixes = [
        "what is ",
        "what are ",
        "define ",
        "explain what ",
    ]

    return any(
        text.startswith(prefix)
        for prefix in prefixes
    )


def _looks_corrupted(
    response: str,
) -> bool:

    # Detect very long joined words / malformed generation
    words = re.findall(
        r"\S+",
        response
    )

    if any(
        len(word) > 40
        for word in words
    ):
        return True

    if len(response.strip()) < 10:
        return True

    return False


def _tool_fallback_answer(
    user_message: str,
    tool_results: list,
) -> str:

    if not tool_results:

        return (
            "The available diagnostic information "
            "is not sufficient to provide a reliable answer."
        )

    runbook = tool_results[0]

    symptom = runbook.get(
        "symptom",
        user_message,
    )

    service = runbook.get(
        "service",
        "system",
    )

    steps = runbook.get(
        "steps",
        [],
    )

    health = {}

    kb_passage = None

    for step in steps:

        if step.get("step") == "health_check":
            health = step.get(
                "result",
                {}
            )

        if step.get("step") == "knowledge_base_search":

            passages = (
                step
                .get("result", {})
                .get("passages", [])
            )

            if passages:
                kb_passage = passages[0]

    status = health.get(
        "status",
        "unknown",
    )

    answer = (
        f"The diagnostic check shows that the "
        f"{service} service is currently {status}. "
    )

    if (
        "connections_pct"
        in health
    ):

        answer += (
            f"Current connection utilization is "
            f"{health['connections_pct']}%. "
        )

    if kb_passage:

        answer += (
            kb_passage.get(
                "text",
                ""
            )
            + " "
        )

    answer += (
        "If the problem continues after these checks, "
        "escalate the issue to human support."
    )

    return answer.strip()


# =========================================================
# Support Model
# =========================================================

def run_support_model(
    user_message: str,
    context: list | None = None,
    tool_results: list | None = None,
) -> str:

    context = context or []
    tool_results = tool_results or []

    extra_information = ""

    if context:

        extra_information += (
            "\n\nTrusted context:\n"
            + json.dumps(
                context,
                ensure_ascii=False,
                indent=2,
            )
        )

    if tool_results:

        extra_information += (
            "\n\nTool results:\n"
            + json.dumps(
                tool_results,
                ensure_ascii=False,
                indent=2,
            )
        )

    messages = [
        {
            "role": "system",
            "content":
                SUPPORT_SYSTEM,
        },
        {
            "role": "user",
            "content":
                user_message
                + extra_information,
        },
    ]

    prompt = (
        support_tokenizer
        .apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    inputs = support_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    with torch.no_grad():

        output = support_model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            pad_token_id=
                support_tokenizer.eos_token_id,
            eos_token_id=
                support_tokenizer.eos_token_id,
        )

    new_tokens = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    response = (
        support_tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )
        .strip()
    )


    # =====================================================
    # Guardrail 1 — Factual questions must stay grounded
    # =====================================================

    if (
        _is_factual_question(
            user_message
        )
        and context
    ):

        best_context = context[0].get(
            "text",
            ""
        )

        if best_context:

            response = best_context


    # =====================================================
    # Guardrail 2 — Replace malformed model output
    # =====================================================

    elif (
        _looks_corrupted(
            response
        )
        and tool_results
    ):

        response = _tool_fallback_answer(
            user_message,
            tool_results,
        )


    # =====================================================
    # Guardrail 3 — Explicit uncertainty
    # =====================================================

    if _needs_uncertainty_guardrail(
        user_message
    ):

        uncertainty_phrases = [
            "cannot confirm",
            "can't confirm",
            "cannot determine",
            "can't determine",
            "not enough information",
            "insufficient information",
        ]

        if not any(
            phrase in response.lower()
            for phrase in uncertainty_phrases
        ):

            response = (
                "I cannot confirm the exact cause "
                "with the available information. "
                + response
            )

    return response