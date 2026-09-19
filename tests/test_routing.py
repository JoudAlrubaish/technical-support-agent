import time
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score

from src.routers import (
    rules_classifier_router,
    llm_router,
    hybrid_router,
)


# =========================================================
# Router Evaluation Set
# =========================================================

TEST_CASES = [
    # QA
    (
        "According to the documentation, which port should be exposed?",
        "qa",
    ),
    (
        "What does the manual say about deployment configuration?",
        "qa",
    ),
    (
        "According to the docs, how should authentication be configured?",
        "qa",
    ),

    # Tools
    (
        "My Wi-Fi keeps disconnecting and I need to diagnose it.",
        "tools",
    ),
    (
        "The API returns 503 after deployment. Check the system status.",
        "tools",
    ),
    (
        "My PostgreSQL connection pool is at 98 percent.",
        "tools",
    ),

    # Support Specialist
    (
        "What is QLoRA?",
        "support_specialist",
    ),
    (
        "Explain why APIs use authentication tokens.",
        "support_specialist",
    ),
    (
        "How can I troubleshoot a slow computer?",
        "support_specialist",
    ),

    # Escalation
    (
        "Production database corruption is suspected.",
        "escalate",
    ),
    (
        "There may be a security breach in production.",
        "escalate",
    ),
    (
        "Critical production system is down and data loss is possible.",
        "escalate",
    ),
]


def evaluate_router(name, router):

    predictions = []
    expected = []
    latencies = []
    json_valid = []

    for text, target in TEST_CASES:

        result = router(text)

        expected.append(target)
        predictions.append(result["route"])
        latencies.append(result.get("latency_ms", 0))

        if name == "LLM":
            json_valid.append(
                result.get("valid_json", False)
            )

        print(
            f"{name:20} | "
            f"Expected: {target:20} | "
            f"Predicted: {result['route']}"
        )

    accuracy = accuracy_score(
        expected,
        predictions,
    )

    macro_f1 = f1_score(
        expected,
        predictions,
        average="macro",
    )

    result = {
        "Router": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Avg Latency (ms)": sum(latencies) / len(latencies),
    }

    if name == "LLM":
        result["JSON Valid Rate"] = (
            sum(json_valid) / len(json_valid)
        )

    if name == "Hybrid":

        fallback_count = 0

        for text, _ in TEST_CASES:

            decision = hybrid_router(text)

            if decision["source"] == "hybrid_llm":
                fallback_count += 1

        result["Fallback Rate"] = (
            fallback_count / len(TEST_CASES)
        )

    return result


# =========================================================
# Run Comparison
# =========================================================

results = []

results.append(
    evaluate_router(
        "Rules + Classifier",
        rules_classifier_router,
    )
)

results.append(
    evaluate_router(
        "LLM",
        llm_router,
    )
)

results.append(
    evaluate_router(
        "Hybrid",
        hybrid_router,
    )
)


results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("FINAL ROUTER COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


#to choose the best threshold for the classifier, we can print out the confidence scores for each test case and see how they align with the expected routes.

from src.routers import classifier_route, rule_first

print("\nCLASSIFIER CONFIDENCES")
print("=" * 70)

for text, expected in TEST_CASES:
    if rule_first(text):
        continue

    result = classifier_route(text)

    print(
        f"{expected:20} | "
        f"{result['intent']:15} | "
        f"{result['confidence']:.4f} | "
        f"{text}"
    )