# Cost-Aware AI Support Triage API

FastAPI implementation of a Triage API with DigitalOcean Serveless Inference's Inference Router. 

The API exposes `POST /triage` and routes each support workflow step through a DigitalOcean Inference Router task:

| Task | Purpose | Suggested router policy |
| --- | --- | --- |
| `classify_ticket` | Categorize the ticket | Lowest cost |
| `urgency_detection` | Score severity, sentiment, and escalation risk | Lowest latency |
| `draft_customer_reply` | Draft a customer-facing response | Higher-quality pool |
| `escalate_complex_issue` | Create a human-agent brief when escalation is needed | Higher-quality pool |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your DigitalOcean model access key and router ID:

```bash
DO_INFERENCE_BASE_URL=https://inference.do-ai.run
DO_MODEL_ACCESS_KEY=your_model_access_key_here
DO_ROUTER_ID=your_router_id_here
TRIAGE_USE_MOCKS=false
```

For local testing without credentials, leave `TRIAGE_USE_MOCKS=true`.

## Run

```bash
uvicorn main:app --reload
```

Open the docs at `http://127.0.0.1:8000/docs`.

## Try A Ticket

```bash
curl -X POST http://127.0.0.1:8000/triage \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Production outage",
    "body": "Our team has been unable to access the dashboard since 09:14 UTC. Around 200 internal users are blocked. Logs show 502s from the API gateway.",
    "customer_tier": "enterprise"
  }'
```

## Try A Batch

Send the included realistic ticket mix through the router:

```bash
curl -X POST http://127.0.0.1:8000/triage/batch \
  -H "Content-Type: application/json" \
  --data-binary @sample_tickets.json
```

Or print a readable command-line report:

```bash
python3 run_batch.py sample_tickets.json
```

For the full raw response, including every route object:

```bash
python3 run_batch.py sample_tickets.json --json
```

## Router Configuration

Create a DigitalOcean Inference Router with these task names exactly:

```text
classify_ticket
urgency_detection
draft_customer_reply
escalate_complex_issue
```

The app sends task-specific system prompts to the router. The router ID stays in `DO_ROUTER_ID`, and the underlying model names and pool-selection logic stay out of the API handler.
