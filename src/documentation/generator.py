"""
DocumentationGenerator: uses Sonnet to auto-generate technical documentation
from pipeline results — methodology, findings, and a project README.
"""

from __future__ import annotations

import os
from typing import Optional

import anthropic

_MODEL = "claude-sonnet-4-6"

_SYSTEM = """\
You are a technical writer specialising in data science and supply chain projects.
Write clear, precise documentation aimed at a technical recruiter or internship evaluator.
Use Markdown. Be concrete: cite numbers from the data, name the models and libraries used.
Do not pad with filler sentences.
"""


class DocumentationGenerator:
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set.")
        self.client = anthropic.Anthropic(api_key=key)

    def _call(self, prompt: str, max_tokens: int = 1500) -> str:
        r = self.client.messages.create(
            model=_MODEL,
            max_tokens=max_tokens,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text.strip()

    # ── Section generators ────────────────────────────────────────────

    def methodology(self, context: str) -> str:
        return self._call(f"""
Write a **Methodology** section (300-400 words) for a supply chain AI project.
Cover: data preparation, forecasting approach (models tried, selection criterion),
optimisation formulation (LP variables, objective, constraints), sensitivity analysis,
and LLM integration points.
Be specific about libraries (statsmodels, PuLP, Anthropic SDK) and model names.

Pipeline context:
{context}
""", max_tokens=700)

    def results_summary(self, context: str) -> str:
        return self._call(f"""
Write a **Results & Key Findings** section (250-350 words).
Structure: forecast accuracy, optimisation outcome, risk analysis (at-risk weeks,
sensitivity breaking points), and what the LLM layer adds.
Quote the actual numbers from the context below.

Pipeline context:
{context}
""", max_tokens=600)

    def architecture_overview(self) -> str:
        return self._call("""
Write an **Architecture Overview** section (200-250 words) describing the 5-module pipeline:
1. Ingestion & schema detection
2. Demand forecasting
3. Resource optimisation
4. LLM interpretation & scenario generation
5. Interactive dashboard

For each module name the key technology used and its role in the pipeline.
The project uses: Python, pandas, statsmodels (Holt-Winters ETS + SARIMA),
PuLP (LP/MIP solver), Anthropic Claude API (Haiku for extraction, Sonnet for reasoning),
Streamlit + Plotly for the dashboard.
""", max_tokens=500)

    def readme(self, context: str) -> str:
        return self._call(f"""
Write a complete GitHub **README.md** for this project (600-800 words).

Sections to include:
# AI Supply Chain Copilot
## Problem statement
## Architecture (5-module pipeline, one bullet per module)
## Tech stack (table: module → library/model)
## Quick start (installation + how to run dashboard + how to run each script)
## Key results (cite numbers from the context)
## Project structure (file tree, one line per file)
## Skills demonstrated (link each module to a supply chain / AI skill)

The project is a portfolio piece for an "AI & Supply Chain Optimization Intern" application.

Pipeline context:
{context}
""", max_tokens=1500)

    def generate_all(self, context: str) -> dict[str, str]:
        """Generate all documentation sections and return as a dict."""
        print("  Generating architecture overview...")
        arch = self.architecture_overview()
        print("  Generating methodology...")
        meth = self.methodology(context)
        print("  Generating results summary...")
        results = self.results_summary(context)
        print("  Generating README...")
        readme = self.readme(context)
        return {
            "architecture": arch,
            "methodology": meth,
            "results": results,
            "readme": readme,
        }
