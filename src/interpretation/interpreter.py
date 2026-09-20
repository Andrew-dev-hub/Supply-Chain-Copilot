"""
SupplyChainInterpreter: wraps the Claude API to provide
  1. executive_summary()    — business-language narrative of all results
  2. answer()               — Q&A with optional pipeline re-run for what-if questions
  3. generate_scenarios()   — LLM proposes edge-case test scenarios as structured JSON
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import anthropic

_SYSTEM_PROMPT = """\
You are a senior supply chain planning advisor. You receive structured data from a
demand forecasting and workforce optimisation pipeline and communicate findings
to a non-technical S&OP manager.

Rules:
- Be concise and direct. Use bullet points where appropriate.
- Quantify every claim with numbers from the data.
- Highlight risks first, then opportunities.
- Never invent numbers not present in the data.
- Write in English unless asked otherwise.
- Avoid jargon; prefer plain business language.
"""


class SupplyChainInterpreter:
    """
    Wrapper around the Claude API for interpreting supply chain pipeline outputs.

    Parameters
    ----------
    api_key   : override ANTHROPIC_API_KEY env var
    model     : Claude model to use
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
    ):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Set the env var or pass api_key=."
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model

    # ------------------------------------------------------------------ #
    def _call(self, user_message: str, max_tokens: int = 1024) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text.strip()

    # ------------------------------------------------------------------ #
    def executive_summary(self, context: str) -> str:
        prompt = f"""
Based on the supply chain pipeline results below, write a concise executive summary
(aim for 250-350 words) structured as:
1. **Situation** – what the data shows (forecast accuracy, plan performance)
2. **Key risks** – at-risk weeks, sensitivity breaking points
3. **Recommendations** – 3 concrete, numbered action items
4. **Outlook** – one sentence on overall plan robustness

Pipeline data:
{context}
"""
        return self._call(prompt, max_tokens=600)

    # ------------------------------------------------------------------ #
    def answer(
        self,
        context: str,
        question: str,
        rerun_fn=None,
    ) -> str:
        """
        Answer a natural-language question about the pipeline results.

        If the question is a what-if that requires a pipeline re-run,
        call rerun_fn(axis, delta) and incorporate the new numbers.
        rerun_fn(axis, delta) -> dict with keys: cost, service_level, avg_workers, status
        """
        # Step 1: classify the question
        classify_prompt = f"""
You are given this supply chain data:
{context}

The user asks: "{question}"

Respond ONLY with a JSON object (no markdown fences) in one of these two shapes:

Shape A — factual question (answerable from the data above):
{{"type": "factual", "answer": "<your answer in 2-4 sentences>"}}

Shape B — what-if requiring a pipeline re-run:
{{"type": "what_if", "axis": "<demand|budget|headcount>", "delta": <float>,
  "description": "<one sentence describing the scenario>"}}

For shape B:
- axis "demand"    → delta is a fraction change, e.g. +0.20 for +20% demand
- axis "budget"    → delta is a budget multiplier, e.g. 0.80 for 80% of current budget
- axis "headcount" → delta is number of workers to REMOVE (integer ≥ 0)
"""
        raw = self._call(classify_prompt, max_tokens=300)
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return self._call(
                f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely.",
                max_tokens=400,
            )

        if parsed.get("type") == "factual":
            return parsed.get("answer", "")

        # What-if branch
        axis = parsed.get("axis", "demand")
        delta = float(parsed.get("delta", 0.0))
        description = parsed.get("description", "")

        rerun_result = None
        if rerun_fn is not None:
            try:
                rerun_result = rerun_fn(axis, delta)
            except Exception as e:
                rerun_result = {"error": str(e)}

        if rerun_result and "error" not in rerun_result:
            result_block = (
                f"Re-run result for '{description}':\n"
                f"  Cost: {rerun_result['cost']:,.0f} EUR\n"
                f"  Service level: {rerun_result['service_level']:.1%}\n"
                f"  Avg workers: {rerun_result['avg_workers']:.1f}\n"
                f"  Solver: {rerun_result['status']}"
            )
        else:
            result_block = f"(Pipeline re-run not available: {rerun_result})"

        interpret_prompt = f"""
Context:
{context}

The user asked: "{question}"
This was interpreted as a what-if scenario: {description}

{result_block}

Write a 3-5 sentence business answer comparing this scenario to the baseline,
highlighting the key change in cost, service level, and headcount.
"""
        return self._call(interpret_prompt, max_tokens=400)

    # ------------------------------------------------------------------ #
    def generate_scenarios(self, context: str, n: int = 5) -> list[dict]:
        """
        Ask Claude to propose n edge-case test scenarios as a JSON list.

        Each scenario has: name, axis, delta, description, expected_risk (low/medium/high)
        """
        prompt = f"""
Based on this supply chain data:
{context}

Propose exactly {n} edge-case test scenarios that stress-test the current plan.
Cover a mix of demand shocks, budget constraints, and headcount issues.
Focus on scenarios that are plausible given the data (e.g. seasonal spikes,
budget freezes, absenteeism).

Return ONLY a JSON array (no markdown) with {n} objects, each with:
{{
  "name": "<short scenario name>",
  "axis": "<demand|budget|headcount>",
  "delta": <float>,
  "description": "<one sentence explaining the scenario>",
  "expected_risk": "<low|medium|high>"
}}

For axis values:
- "demand"    → delta is fraction change (e.g. 0.35 for +35% demand shock)
- "budget"    → delta is budget multiplier (e.g. 0.75 for 75% of current budget)
- "headcount" → delta is integer number of workers to remove (e.g. 5)
"""
        raw = self._call(prompt, max_tokens=800)
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
