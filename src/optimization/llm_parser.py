"""
LLM-driven constraint parser: natural language → WorkforceConstraints.

Uses the Claude API (claude-sonnet-4-6) to extract structured parameters
from a plain-English description of planning constraints.
Requires ANTHROPIC_API_KEY environment variable.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from .constraints import WorkforceConstraints

_SYSTEM_PROMPT = """\
You are a supply chain planning assistant. Your task is to extract workforce scheduling
parameters from a natural-language description and return them as a JSON object.

Return ONLY a valid JSON object with these keys (use null for missing optional fields):
{
  "capacity_per_worker": <float, units per worker per week>,
  "cost_per_worker": <float, cost in local currency per worker per week>,
  "max_workers": <int, maximum available headcount per week>,
  "budget_total": <float, total labour budget for the whole planning horizon>,
  "min_service_level": <float, fraction 0-1, default 0.90 if not specified>,
  "unmet_demand_penalty": <float, default 5000 if not specified>,
  "max_ramp_up": <int or null>,
  "max_ramp_down": <int or null>,
  "currency": <string, e.g. "EUR" or "USD", default "EUR">
}

Rules:
- If a daily capacity is given, multiply by 5 to get weekly capacity.
- If a monthly budget is given, convert to the planning horizon length if known,
  otherwise use the monthly value as-is and note this in a "warning" field.
- If a field is absent and has no default, use null (the caller will ask the user).
- Do not include explanations — only the JSON object.
"""


def parse_constraints_from_text(
    text: str,
    horizon_weeks: Optional[int] = None,
    api_key: Optional[str] = None,
) -> WorkforceConstraints:
    """
    Send *text* to Claude and parse the returned JSON into a WorkforceConstraints.

    Parameters
    ----------
    text           : natural-language description of constraints
    horizon_weeks  : length of planning horizon (helps LLM scale monthly budgets)
    api_key        : override ANTHROPIC_API_KEY env var
    """
    import anthropic

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Either set the env var or pass api_key=."
        )

    client = anthropic.Anthropic(api_key=key)

    user_message = text
    if horizon_weeks:
        user_message = f"Planning horizon: {horizon_weeks} weeks.\n\n{text}"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\n\nRaw response:\n{raw}") from e

    # Validate required fields
    missing = [
        k for k in ("capacity_per_worker", "cost_per_worker", "max_workers", "budget_total")
        if data.get(k) is None
    ]
    if missing:
        raise ValueError(
            f"LLM could not extract required fields: {missing}. "
            "Please provide more detail in your constraint description."
        )

    return WorkforceConstraints(
        capacity_per_worker=float(data["capacity_per_worker"]),
        cost_per_worker=float(data["cost_per_worker"]),
        max_workers=int(data["max_workers"]),
        budget_total=float(data["budget_total"]),
        min_service_level=float(data.get("min_service_level") or 0.90),
        unmet_demand_penalty=float(data.get("unmet_demand_penalty") or 5000.0),
        max_ramp_up=int(data["max_ramp_up"]) if data.get("max_ramp_up") is not None else None,
        max_ramp_down=int(data["max_ramp_down"]) if data.get("max_ramp_down") is not None else None,
        currency=str(data.get("currency") or "EUR"),
    )
