import json
import os
import re
import time
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field


load_dotenv()

DO_INFERENCE_BASE_URL = os.getenv("DO_INFERENCE_BASE_URL", "https://inference.do-ai.run")
DO_MODEL_ACCESS_KEY = os.getenv("DO_MODEL_ACCESS_KEY")
DO_ROUTER_ID = os.getenv("DO_ROUTER_ID")
USE_MOCKS = os.getenv("TRIAGE_USE_MOCKS", "").lower() in {"1", "true", "yes", "on"}

client = OpenAI(
    base_url=DO_INFERENCE_BASE_URL,
    api_key=DO_MODEL_ACCESS_KEY or "missing-key-for-local-mock-mode",
)

app = FastAPI(
    title="Cost-Aware AI Support Triage API",
    version="1.0.0",
    description="Routes support triage subtasks through the DigitalOcean Inference Router.",
)


Category = Literal["billing", "bug", "how-to", "account", "feature-request", "other"]


class Ticket(BaseModel):
    subject: str = Field(..., min_length=1, max_length=200)
    body: str = Field(..., min_length=1)
    customer_tier: str | None = Field(default=None, max_length=80)
    account_id: str | None = Field(default=None, max_length=120)


class UrgencyResult(BaseModel):
    score: int = Field(..., ge=1, le=5)
    sentiment: str
    escalation_risk: bool
    reason: str


class RouteMetadata(BaseModel):
    task: str
    latency_ms: int
    mode: Literal["mock", "router"]
    selected_model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class TriageResponse(BaseModel):
    category: Category
    urgency: UrgencyResult
    reply: str
    escalation_summary: str | None
    routes: list[RouteMetadata]


class BatchTriageResponse(BaseModel):
    count: int
    results: list[TriageResponse | dict[str, Any]]


TASK_PROMPTS = {
    "classify_ticket": (
        "Classify this support ticket into exactly one category: billing, bug, "
        "how-to, account, feature-request, other. Reply with JSON only: "
        '{"category":"billing|bug|how-to|account|feature-request|other"}'
    ),
    "urgency_detection": (
        "Score urgency from 1 (low) to 5 (critical), identify sentiment, and decide "
        "whether there is escalation risk. Reply with JSON only: "
        '{"score":1,"sentiment":"neutral","escalation_risk":false,"reason":"short reason"}'
    ),
    "draft_customer_reply": (
        "Write a short, professional customer-facing reply. Acknowledge the issue, "
        "set expectations, and keep it to four sentences or fewer. Reply with JSON only: "
        '{"reply":"message"}'
    ),
    "escalate_complex_issue": (
        "Summarize this support ticket for a human agent. Include the problem, "
        "customer impact, what has been tried, and recommended next steps. Reply with JSON only: "
        '{"summary":"structured brief"}'
    ),
}


def has_real_setting(value: str | None, placeholder: str) -> bool:
    return bool(value and value.strip() and value != placeholder)


def ticket_text(ticket: Ticket) -> str:
    context = []
    if ticket.customer_tier:
        context.append(f"Customer tier: {ticket.customer_tier}")
    if ticket.account_id:
        context.append(f"Account ID: {ticket.account_id}")

    context_block = "\n".join(context)
    if context_block:
        context_block = f"{context_block}\n\n"

    return f"{context_block}Subject: {ticket.subject}\n\n{ticket.body}"


def parse_json_object(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise ValueError("Model did not return a JSON object") from None
        return json.loads(match.group(0))


def selected_model_from_response(response: Any) -> str | None:
    for attr in ("model", "selected_model"):
        value = getattr(response, attr, None)
        if value:
            return str(value)

    metadata = getattr(response, "metadata", None)
    if isinstance(metadata, dict):
        value = metadata.get("selected_model") or metadata.get("model")
        if value:
            return str(value)

    return None


def usage_from_response(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def call_router(task: str, text: str) -> tuple[dict[str, Any], RouteMetadata]:
    if USE_MOCKS:
        started = time.perf_counter()
        result = mock_task(task, text)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return result, RouteMetadata(task=task, latency_ms=latency_ms, mode="mock")

    if not has_real_setting(DO_MODEL_ACCESS_KEY, "your_model_access_key_here") or not has_real_setting(
        DO_ROUTER_ID, "your_router_id_here"
    ):
        raise HTTPException(
            status_code=500,
            detail=(
                "Set DO_MODEL_ACCESS_KEY and DO_ROUTER_ID, or set TRIAGE_USE_MOCKS=true "
                "to run the local demo without DigitalOcean credentials."
            ),
        )

    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=DO_ROUTER_ID,
            messages=[
                {"role": "system", "content": TASK_PROMPTS[task]},
                {"role": "user", "content": text},
            ],
        )
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"DigitalOcean Inference request failed: {exc}") from exc
    latency_ms = int((time.perf_counter() - started) * 1000)
    content = response.choices[0].message.content or "{}"

    route = RouteMetadata(
        task=task,
        latency_ms=latency_ms,
        mode="router",
        selected_model=selected_model_from_response(response),
        **usage_from_response(response),
    )
    return parse_json_object(content), route


def mock_task(task: str, text: str) -> dict[str, Any]:
    lower = text.lower()
    is_enterprise = any(word in lower for word in ["enterprise", "production", "outage", "blocked"])
    is_angry = any(word in lower for word in ["ridiculous", "angry", "third time", "alternatives", "churn"])
    has_logs = any(word in lower for word in ["502", "logs", "trace", "gateway", "stack"])

    if task == "classify_ticket":
        if any(word in lower for word in ["invoice", "charged", "billing", "payment"]):
            return {"category": "billing"}
        if any(word in lower for word in ["password", "login", "401", "auth", "key"]):
            return {"category": "account"}
        if "feature" in lower or "bulk export" in lower:
            return {"category": "feature-request"}
        if any(word in lower for word in ["bug", "crash", "502", "down", "weird", "error", "outage"]):
            return {"category": "bug"}
        if any(word in lower for word in ["how do i", "how to", "docs"]):
            return {"category": "how-to"}
        return {"category": "other"}

    if task == "urgency_detection":
        score = 5 if is_enterprise or "outage" in lower else 4 if is_angry else 3 if has_logs else 2
        sentiment = "angry" if is_angry else "worried" if score >= 4 else "neutral"
        return {
            "score": score,
            "sentiment": sentiment,
            "escalation_risk": score >= 4 or has_logs,
            "reason": "High customer impact or churn risk detected." if score >= 4 else "Limited impact described.",
        }

    if task == "draft_customer_reply":
        return {
            "reply": (
                "Thanks for reaching out. We understand the impact this is having and will "
                "review the details you shared right away. Our team will follow up with the "
                "next step or a clarifying question shortly."
            )
        }

    if task == "escalate_complex_issue":
        return {
            "summary": (
                "Problem: customer reports a support issue that may require human review. "
                "Impact: urgency indicators suggest possible business disruption or churn risk. "
                "Evidence: ticket includes operational details, logs, repeated failures, or account context. "
                "Recommended next steps: validate account state, inspect recent errors, and send a timely update."
            )
        }

    raise ValueError(f"Unknown task: {task}")


@app.get("/health")
def health() -> dict[str, str | bool | None]:
    return {
        "status": "ok",
        "mode": "mock" if USE_MOCKS else "router",
        "base_url": DO_INFERENCE_BASE_URL,
        "router_configured": has_real_setting(DO_ROUTER_ID, "your_router_id_here"),
    }


@app.post("/triage", response_model=TriageResponse)
def triage(ticket: Ticket) -> TriageResponse:
    text = ticket_text(ticket)
    routes: list[RouteMetadata] = []

    category_payload, category_route = call_router("classify_ticket", text)
    routes.append(category_route)

    urgency_payload, urgency_route = call_router("urgency_detection", text)
    routes.append(urgency_route)
    urgency = UrgencyResult.model_validate(urgency_payload)

    reply_payload, reply_route = call_router("draft_customer_reply", text)
    routes.append(reply_route)

    escalation_summary = None
    if urgency.escalation_risk or urgency.score >= 4:
        escalation_payload, escalation_route = call_router("escalate_complex_issue", text)
        routes.append(escalation_route)
        escalation_summary = str(escalation_payload["summary"])

    return TriageResponse(
        category=category_payload["category"],
        urgency=urgency,
        reply=str(reply_payload["reply"]),
        escalation_summary=escalation_summary,
        routes=routes,
    )


@app.post("/triage/batch", response_model=BatchTriageResponse)
def triage_batch(tickets: list[Ticket]) -> BatchTriageResponse:
    if not tickets:
        raise HTTPException(status_code=400, detail="Send at least one ticket.")
    if len(tickets) > 25:
        raise HTTPException(status_code=400, detail="Batch size is limited to 25 tickets.")

    results: list[TriageResponse | dict[str, Any]] = []
    for index, ticket in enumerate(tickets):
        try:
            results.append(triage(ticket))
        except Exception as exc:
            results.append(
                {
                    "ticket_index": index,
                    "subject": ticket.subject,
                    "error": str(exc),
                }
            )

    return BatchTriageResponse(count=len(tickets), results=results)
