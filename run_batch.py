import argparse
import json
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def post_json(url: str, payload: Any) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        raise SystemExit(f"Request failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach the API at {url}: {exc.reason}") from exc


def compact_routes(routes: list[dict[str, Any]]) -> str:
    parts = []
    for route in routes:
        model = route.get("selected_model") or route.get("mode")
        parts.append(f"{route['task']}={model} ({route['latency_ms']}ms)")
    return "; ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of support tickets through /triage/batch.")
    parser.add_argument("tickets", nargs="?", default="sample_tickets.json", help="Path to a JSON ticket array.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/triage/batch", help="Batch endpoint URL.")
    parser.add_argument("--json", action="store_true", help="Print the raw JSON response.")
    args = parser.parse_args()

    tickets = json.loads(Path(args.tickets).read_text())
    response = post_json(args.url, tickets)

    if args.json:
        print(json.dumps(response, indent=2))
        return

    for index, (ticket, result) in enumerate(zip(tickets, response["results"], strict=True), start=1):
        if "error" in result:
            print(f"{index}. {ticket['subject']}")
            print(f"   error: {result['error']}")
            print()
            continue

        urgency = result["urgency"]
        print(f"{index}. {ticket['subject']}")
        print(f"   category: {result['category']}")
        print(
            "   urgency: "
            f"{urgency['score']}/5, sentiment={urgency['sentiment']}, "
            f"escalation={urgency['escalation_risk']}"
        )
        print(f"   reply: {textwrap.shorten(result['reply'], width=130, placeholder='...')}")
        if result["escalation_summary"]:
            print(f"   escalation: {textwrap.shorten(result['escalation_summary'], width=130, placeholder='...')}")
        print(f"   routes: {compact_routes(result['routes'])}")
        print()


if __name__ == "__main__":
    main()
