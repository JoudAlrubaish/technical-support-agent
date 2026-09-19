# Design Decisions

This document records the main engineering decisions behind the Multi-Model Agentic Technical Support System.

## 1. Why Three Models

The system intentionally separates three machine-learning tasks instead of depending on one model for every operation.

### Model A — Intent Classifier

Base model:

```text
distilbert-base-uncased
```

Classes:

```text
authentication
network
deployment
database
gpu
api
package
general
```

Purpose:

Model A provides a fast and inexpensive routing signal before using a generative model.

Final test performance:

```text
Accuracy: 0.95
Macro F1: 0.9495
Minimum class recall: 0.80
```

The model passed the defined classifier quality gate.

---

### Model B — Extractive QA

Base model:

```text
distilbert-base-uncased
```

Purpose:

Model B is used for questions that should be answered directly from known technical documentation.

It extracts an answer span from retrieved context rather than freely generating the answer.

Final result:

```text
Exact Match: 1.00
Token F1: 1.00
```

The final test set includes only eight QA pairs, so the perfect score should be interpreted carefully.

---

### Model C — Support Specialist

Base model:

```text
HuggingFaceTB/SmolLM2-135M-Instruct
```

Training approach:

```text
Supervised Fine-Tuning
LoRA
```

Final selected experiment:

```text
Exp 6
```

Final result:

```text
Validation Loss: 1.0784
Perplexity: 2.9399
ROUGE-L: 0.4422
Golden Set: 83.33%
```

Model C is used for:

```text
technical-support reasoning
tool-result synthesis
general troubleshooting
ambiguous router fallback
```

---

## 2. Model C LoRA Configuration

The main LoRA configuration was:

```text
r = 16
lora_alpha = 32
lora_dropout = 0.05
```

Target modules:

```text
q_proj
k_proj
v_proj
o_proj
```

Task type:

```text
CAUSAL_LM
```

This provides more adaptation capacity than targeting only Q and V projections while still training a small number of parameters compared with full fine-tuning.

---

## 3. Model C Experiment Strategy

The first experiments focused on improving general model performance.

### Baseline

```text
Validation Loss: 3.2272
Perplexity: 25.2089
ROUGE-L: 0.1363
```

### Exp 1

```text
Epochs: 5
Validation Loss: 2.3036
Perplexity: 10.0102
```

The model was still underfitting.

### Exp 2

```text
Epochs: 10
Validation Loss: 1.4185
Perplexity: 4.1311
ROUGE-L: 0.1962
```

### Exp 3

```text
Epochs: 15
Validation Loss: 1.0825
Perplexity: 2.9521
ROUGE-L: 0.4322
```

General performance became strong, but behavioral testing exposed weaknesses.

Golden Set result:

```text
3 / 6
50%
```

---

## 4. Why Behavioral Fine-Tuning Was Added

Traditional metrics did not fully represent production behavior.

For example, a model can have:

```text
low validation loss
low perplexity
higher ROUGE-L
```

while still failing important behaviors such as:

```text
uncertainty
security escalation
tool-result synthesis
```

Therefore a Golden Set was added.

---

## 5. Golden Set

The Golden Set contains six behavior categories:

```text
troubleshooting
network
uncertainty
escalation
tool_synthesis
security_escalation
```

Exp 3:

```text
50.00%
```

Exp 4:

```text
66.67%
```

Exp 5:

```text
66.67%
```

Exp 6:

```text
83.33%
```

Exp 6 was selected because it achieved the best overall balance.

---

## 6. Why Exp 6 Was Selected

Exp 4 and Exp 5 showed that highly targeted training could improve one behavior while causing regression in another.

For example:

```text
uncertainty improved
but tool synthesis regressed
```

Exp 6 therefore restarted from the clean Exp 3 checkpoint and used balanced behavioral replay.

Exp 6 configuration:

```text
Epochs: 3
Learning Rate: 3e-5
Training Examples: 18
Max Length: 512
Seed: 42
```

The 18 examples combined:

```text
12 replay examples

+

targeted behavioral examples
```

Replay behaviors included:

```text
troubleshooting
tool_synthesis
escalation
uncertainty
```

The selected model reached:

```text
Validation Loss: 1.0784
Perplexity: 2.9399
ROUGE-L: 0.4422
Golden Set: 83.33%
```

---

## 7. Router Strategy

Three router strategies were evaluated.

| Strategy | Accuracy | Macro F1 | Average Latency |
|---|---:|---:|---:|
| Rules + Classifier | 0.9167 | 0.9143 | 27.98 ms |
| LLM Router | 0.2500 | 0.1154 | 802.32 ms |
| Hybrid Router | 1.0000 | 1.0000 | 137.49 ms |

These results are based on the current 12-case router evaluation set.

They should not be interpreted as universal production accuracy.

---

## 8. Why the Standalone LLM Router Was Not Selected

The standalone LLM router successfully produced valid JSON, but its routing accuracy was poor on the evaluation set.

Result:

```text
Accuracy: 25%
Macro F1: 0.1154
Latency: 802.32 ms
```

Therefore, using the LLM for every routing request would be:

```text
slower
less accurate on the test set
more computationally expensive
```

The final system only uses the LLM router when the classifier is uncertain.

---

## 9. Final Hybrid Routing Policy

The final routing order is:

```text
Hard Rules
↓
Model A Intent Classifier
↓
Confidence Check
↓
LLM Router only when ambiguous
```

Classifier confidence threshold:

```text
0.25
```

Logic:

```text
if hard rule matches:
    use hard rule

else:
    run Model A

if classifier confidence >= 0.25:
    use classifier route

else:
    use Model C LLM router
```

Hybrid fallback rate on the router test set:

```text
16.67%
```

---

## 10. Why Hard Rules Run First

Some cases should never depend entirely on probabilistic ML routing.

Examples:

```text
data loss
database corruption
security breach
production down
```

These conditions directly trigger:

```text
human escalation
```

Similarly, explicit documentation questions directly trigger:

```text
QA route
```

This makes critical routing deterministic.

---

## 11. When QA Wins

The QA path has priority when the request clearly refers to technical documentation.

Examples:

```text
according to the documentation
according to the docs
documentation
manual
according to the deployment guide
```

The reason is grounding.

If the answer is available in documentation, extracting the answer using Model B is safer than asking Model C to recreate the fact.

---

## 12. Intent-to-Route Mapping

Classifier labels are mapped to agent routes as follows:

```text
authentication -> support_specialist

network        -> tools
deployment     -> tools
database       -> tools
gpu            -> tools
api            -> tools
package        -> tools

general        -> support_specialist
```

The predicted intent and final route are different concepts.

For example:

```text
intent = database
route = tools
```

because database incidents often need diagnostics before generation.

---

## 13. Tool Design

The project exposes all 13 required tool contracts.

```text
knowledge_base_search
ticket_search
ticket_create
system_health_check
log_analyzer
documentation_search
package_lookup
sql_query
calculator
file_search
web_search
escalate_to_human
diagnostic_runbook
```

Tool outputs use structured dictionaries rather than uncontrolled text.

---

## 14. SQL Safety

The SQL tool is intentionally read-only.

Allowed:

```sql
SELECT ...
```

Not allowed:

```text
INSERT
UPDATE
DELETE
DROP
ALTER
```

This prevents the autonomous agent from modifying support data through arbitrary generated SQL.

---

## 15. Diagnostic Runbook

The diagnostic runbook combines several deterministic checks.

Typical sequence:

```text
system health check
↓
ticket search
↓
knowledge base search
↓
structured diagnostic result
```

The structured result can then be passed to Model C for user-friendly synthesis.

This separates:

```text
deterministic system inspection
```

from:

```text
natural-language explanation
```

---

## 16. LangGraph Orchestration

LangGraph controls the system state and execution flow.

Main nodes:

```text
route_node
qa_node
tool_node
support_node
escalation_node
```

The graph keeps track of:

```text
user_message
route
router_source
intent
confidence
context
tool_results
answer
escalated
trace_id
router_latency_ms
```

---

## 17. Tool-to-Support Flow

The tools route does not immediately return raw diagnostic dictionaries to the user.

Instead:

```text
tool_node
↓
support_node
```

Model C synthesizes the structured diagnostic information into a readable technical-support response.

This allows deterministic tools to handle facts while the LLM handles explanation.

---

## 18. Escalation Flow

High-risk incidents terminate through:

```text
escalation_node
```

The escalation tool creates a support ticket and returns a response such as:

```text
This issue has been escalated to human support.
Ticket ID: ...
```

The system avoids continuing automated troubleshooting for dangerous cases.

---

## 19. Model C Grounding Problem

During integration testing, Model C sometimes produced hallucinated or malformed technical facts.

One important regression example was:

```text
What is QLoRA?
```

Without grounding, the model could invent an incorrect meaning.

The system therefore does not trust Model C generation alone for factual questions.

---

## 20. Factual Grounding Guardrail

When a request is a factual question and useful retrieved context exists, the system returns the grounded information directly.

For example:

```text
QLoRA is a parameter-efficient fine-tuning method for large language models.
It keeps the pretrained model quantized to 4-bit and trains LoRA adapters.
```

This prevents the generative model from changing a known fact.

---

## 21. Explicit Uncertainty Guardrail

Model C's final Golden Set weakness was uncertainty handling.

When a user asks for an exact root cause without enough evidence, the system adds:

```text
I cannot confirm the exact cause with the available information.
```

before providing troubleshooting guidance.

This guardrail is deterministic.

---

## 22. Generation Corruption Guardrail

Model C occasionally produced malformed or low-quality text during early integration.

The production layer detects suspicious output such as:

```text
extremely long malformed words
very short unusable responses
```

When tool results are available, the system falls back to a deterministic response built from those results.

---

## 23. Regression Testing

A regression suite was created after integration failures were discovered.

Current result:

```text
13 / 13 PASS
```

Important regression cases include:

```text
documentation grounding
database diagnostics
QLoRA grounding
security escalation
uncertainty handling
tool response structure
```

Regression tests are important because aggregate ML metrics can improve while a specific production behavior breaks.

---

## 24. Langfuse Observability

Langfuse was integrated through:

```text
src/observability.py
```

The system records:

```text
trace ID
session ID
router metadata
route
router source
LangGraph callbacks
final output
```

Secrets are stored only in environment variables.

They are never committed to GitHub.

---

## 25. API Design

The final system exposes a unified OpenAI-compatible endpoint.

Endpoints:

```text
GET /health
GET /v1/models
POST /v1/chat/completions
```

Unified model ID:

```text
tuwaiq-tech-support-agent
```

The API represents the complete agent rather than exposing Model C directly.

This ensures every request can still use:

```text
router
QA
tools
guardrails
escalation
Langfuse
```

---

## 26. Streaming

Open WebUI sends:

```text
stream=true
```

by default.

Therefore Server-Sent Events streaming support was added to the FastAPI endpoint.

Streaming responses follow the OpenAI-style format:

```text
chat.completion.chunk
```

and finish with:

```text
data: [DONE]
```

---

## 27. Why Open WebUI Connects to the Unified Agent

Open WebUI is connected to:

```text
http://support-agent:8000/v1
```

It does not connect directly to Model C.

This is important because direct connection to Model C would bypass:

```text
routing
tools
QA
safety
human escalation
observability
```

---

## 28. Docker Compose Design

Docker Compose currently contains two services:

```text
support-agent
open-webui
```

Ports:

```text
support-agent:
8000

open-webui:
3001
```

Open WebUI uses `3001` because Dokploy itself occupies port `3000` in the local deployment environment.

---

## 29. Dokploy Deployment

Dokploy was installed locally inside:

```text
Ubuntu 24.04
Multipass VM
```

The repository was connected through a GitHub App.

Deployment configuration:

```text
Repository:
JoudAlrubaish/technical-support-agent

Branch:
main

Compose Path:
./docker-compose.yml

Trigger:
On Push
```

The Docker Compose deployment completed successfully.

---

## 30. Dokploy Validation

The Open WebUI instance deployed through Dokploy successfully called the unified technical-support agent.

Test question:

```text
According to the documentation, which port should be exposed?
```

Returned answer:

```text
Expose port 8000 when deploying the FastAPI service
```

This verifies:

```text
Dokploy
↓
Open WebUI
↓
FastAPI unified endpoint
↓
LangGraph
↓
QA path
↓
Model B / grounded answer
```

---

## 31. Local Deployment Scope

The Dokploy deployment is currently local.

It runs inside:

```text
Multipass Ubuntu VM
```

No external paid VPS was used.

Therefore:

```text
Dokploy local deployment: completed
Public HTTPS deployment: not completed
```

This limitation is documented rather than hidden.

---

## 32. Current Limitations

The current system has the following limitations:

```text
Model B test set contains only 8 QA pairs.

Router evaluation contains only 12 cases.

Model C still needs an uncertainty guardrail.

Several tools use local deterministic/mock backends.

Token usage in API responses is currently reported as zero.

Dokploy is local only.

No public HTTPS domain is configured.
```