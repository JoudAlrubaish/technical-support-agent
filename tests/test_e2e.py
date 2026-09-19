import json
import os

from src.graph import run_agent
from src.tools import system_health_check


RESULTS = []


def check(name, condition, details=""):
    status = "PASS" if condition else "FAIL"

    RESULTS.append({
        "test": name,
        "status": status,
        "details": details,
    })

    print(f"{status:4} | {name}")

    if not condition:
        raise AssertionError(
            f"{name} failed: {details}"
        )


# =========================================================
# 1. Documentation -> QA -> Model B
# =========================================================

result = run_agent(
    "According to the documentation, which port should be exposed?"
)

check(
    "Documentation QA route",
    result["route"] == "qa",
    str(result),
)

check(
    "Documentation grounded answer",
    "8000" in result["answer"],
    result["answer"],
)


# =========================================================
# 2. Tools -> Diagnostic Runbook -> Model C / Guardrail
# =========================================================

result = run_agent(
    "My PostgreSQL connection pool is at 98 percent and requests time out."
)

check(
    "Database issue routes to tools",
    result["route"] == "tools",
    str(result),
)

check(
    "Database diagnostics used",
    len(result.get("tool_results", [])) > 0,
    str(result),
)

check(
    "Database grounded response",
    (
        "degraded" in result["answer"].lower()
        and "91%" in result["answer"]
    ),
    result["answer"],
)


# =========================================================
# 3. General Support -> Model C
# Regression test from discovered QLoRA hallucination
# =========================================================

result = run_agent(
    "What is QLoRA?"
)

check(
    "QLoRA support route",
    result["route"] == "support_specialist",
    str(result),
)

check(
    "QLoRA uses LLM fallback",
    result["router_source"] == "hybrid_llm",
    str(result),
)

check(
    "QLoRA regression grounding",
    (
        "4-bit" in result["answer"].lower()
        and "lora" in result["answer"].lower()
    ),
    result["answer"],
)


# =========================================================
# 4. High-risk Incident -> Human Escalation
# =========================================================

result = run_agent(
    "Production database corruption is suspected."
)

check(
    "Corruption routes to escalation",
    result["route"] == "escalate",
    str(result),
)

check(
    "Human escalation created",
    result.get("escalated") is True,
    str(result),
)

check(
    "Escalation ticket created",
    "Ticket ID" in result["answer"],
    result["answer"],
)


# =========================================================
# 5. Uncertainty Guardrail
# =========================================================

result = run_agent(
    "My computer suddenly became slow. "
    "I have no other information. "
    "Can you tell me the exact cause?"
)

check(
    "Explicit uncertainty behavior",
    "cannot confirm" in result["answer"].lower(),
    result["answer"],
)


# =========================================================
# 6. Tool Contract Check
# =========================================================

health = system_health_check.invoke({
    "service": "api"
})

check(
    "Health tool structured response",
    health["status"]
    in {
        "healthy",
        "degraded",
        "down",
        "unknown",
    },
    str(health),
)


# =========================================================
# Save Results
# =========================================================

os.makedirs(
    "reports",
    exist_ok=True,
)

summary = {
    "total_tests": len(RESULTS),
    "passed": sum(
        x["status"] == "PASS"
        for x in RESULTS
    ),
    "failed": sum(
        x["status"] == "FAIL"
        for x in RESULTS
    ),
    "tests": RESULTS,
}

with open(
    "reports/e2e_regression_results.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )


print("\n" + "=" * 60)
print("E2E + REGRESSION SUMMARY")
print("=" * 60)
print(
    f"Passed: {summary['passed']}/{summary['total_tests']}"
)

print(
    "Saved: reports/e2e_regression_results.json"
)