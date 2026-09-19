import time
import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)


# =========================================================
# Configuration
# =========================================================

MODEL_A_REPO = "JoudAlrubaish/technical-support-intent-classifier"
MODEL_C_REPO = "JoudAlrubaish/technical-support-agent-model-c"

CONFIDENCE_THRESHOLD = 0.25


# =========================================================
# Router A — Rules + Fine-tuned Classifier
# =========================================================

router_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_A_REPO
)

router_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_A_REPO
)

router_model.eval()


RULES = {
    "escalate": [
        "data loss",
        "security breach",
        "production down",
        "corruption",
    ],

    "qa": [
        "according to the docs",
        "according to the documentation",
        "documentation",
        "what does the manual say",
        "according to the deployment guide",
    ],
}


INTENT_TO_ROUTE = {
    "authentication": "support_specialist",
    "network": "tools",
    "deployment": "tools",
    "database": "tools",
    "gpu": "tools",
    "api": "tools",
    "package": "tools",
    "general": "support_specialist",
}


def rule_first(text: str):

    text = text.lower()

    for route, phrases in RULES.items():

        if any(
            phrase in text
            for phrase in phrases
        ):

            return {
                "route": route,
                "source": "rule",
                "confidence": 1.0,
            }

    return None


@torch.no_grad()
def classifier_route(text: str):

    inputs = router_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    logits = router_model(
        **inputs
    ).logits[0]

    probs = torch.softmax(
        logits,
        dim=-1,
    )

    idx = int(
        torch.argmax(probs)
    )

    confidence = float(
        probs[idx]
    )

    intent = router_model.config.id2label[
        idx
    ]

    return {
        "intent": intent,
        "confidence": confidence,
    }


def rules_classifier_router(
    text: str,
):

    start = time.perf_counter()

    rule = rule_first(
        text
    )

    if rule:

        result = rule

    else:

        pred = classifier_route(
            text
        )

        result = {
            "route": INTENT_TO_ROUTE[
                pred["intent"]
            ],
            "intent": pred["intent"],
            "confidence": pred["confidence"],
            "source": "classifier",
        }

    result["latency_ms"] = (
        time.perf_counter() - start
    ) * 1000

    return result


def needs_llm_fallback(
    text: str,
):

    if rule_first(text):
        return False

    pred = classifier_route(
        text
    )

    return (
        pred["confidence"]
        < CONFIDENCE_THRESHOLD
    )


# =========================================================
# Router B — LLM Router
# =========================================================

llm_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_C_REPO
)

llm_router_model = AutoModelForCausalLM.from_pretrained(
    MODEL_C_REPO
)

llm_router_model.eval()


ALLOWED_ROUTES = {
    "qa",
    "support_specialist",
    "tools",
    "escalate",
}


ROUTER_SYSTEM = """
You are a routing controller for a technical support system.

Your only job is to select the correct route.

Return ONLY valid JSON in this exact format:
{"route":"route_name"}

Allowed routes:

qa
Use when the user explicitly asks about trusted documentation,
manuals, guides, or documented information.

tools
Use when the request requires diagnostics, system status,
logs, connectivity checks, packages, APIs, databases,
deployment information, GPU information, or live system data.

support_specialist
Use for general technical explanations, troubleshooting guidance,
or technical questions that do not require live system information.

escalate
Use only for high-risk incidents such as data loss,
security breaches, suspected corruption, critical production
outages, or situations requiring human intervention.

Do not include explanations.
Return JSON only.
"""


def llm_router(
    text: str,
):

    messages = [

        {
            "role": "system",
            "content": ROUTER_SYSTEM,
        },

        {
            "role": "user",
            "content":
                "According to the documentation, "
                "which port should I expose?",
        },

        {
            "role": "assistant",
            "content":
                '{"route":"qa"}',
        },

        {
            "role": "user",
            "content":
                "My Wi-Fi keeps disconnecting and "
                "I need to diagnose the connection.",
        },

        {
            "role": "assistant",
            "content":
                '{"route":"tools"}',
        },

        {
            "role": "user",
            "content":
                "What is QLoRA?",
        },

        {
            "role": "assistant",
            "content":
                '{"route":"support_specialist"}',
        },

        {
            "role": "user",
            "content":
                "Production database corruption is suspected.",
        },

        {
            "role": "assistant",
            "content":
                '{"route":"escalate"}',
        },

        {
            "role": "user",
            "content": text,
        },
    ]

    prompt = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = llm_tokenizer(
        prompt,
        return_tensors="pt",
    )

    start = time.perf_counter()

    with torch.no_grad():

        output = llm_router_model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=
                llm_tokenizer.eos_token_id,
        )

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    generated_tokens = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    raw_output = llm_tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    try:

        parsed = json.loads(
            raw_output
        )

        route = parsed.get(
            "route"
        )

        if route not in ALLOWED_ROUTES:

            raise ValueError(
                f"Invalid route: {route}"
            )

        return {
            "route": route,
            "source": "llm",
            "valid_json": True,
            "latency_ms": latency_ms,
        }

    except Exception:

        return {
            "route": "support_specialist",
            "source": "llm_fallback",
            "valid_json": False,
            "latency_ms": latency_ms,
            "raw_output": raw_output,
        }


# =========================================================
# Router C — Hybrid Router
# =========================================================

def hybrid_router(
    text: str,
):

    start = time.perf_counter()

    # 1. Hard rules
    rule = rule_first(
        text
    )

    if rule:

        rule["latency_ms"] = (
            time.perf_counter() - start
        ) * 1000

        rule["source"] = (
            "hybrid_rule"
        )

        return rule

    # 2. Fine-tuned classifier
    pred = classifier_route(
        text
    )

    if (
        pred["confidence"]
        >= CONFIDENCE_THRESHOLD
    ):

        result = {
            "route": INTENT_TO_ROUTE[
                pred["intent"]
            ],
            "intent": pred["intent"],
            "confidence":
                pred["confidence"],
            "source":
                "hybrid_classifier",
        }

    # 3. LLM fallback
    else:

        llm_result = llm_router(
            text
        )

        result = {
            **llm_result,
            "intent": pred["intent"],
            "classifier_confidence":
                pred["confidence"],
            "source":
                "hybrid_llm",
        }

    result["latency_ms"] = (
        time.perf_counter() - start
    ) * 1000

    return result