"""
AI Supply Chain Copilot — Streamlit Dashboard (Module 5)

Run with:
    py -m streamlit run dashboard.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Cap numeric-library threads before numpy is imported: on a shared host every core a
# model fit grabs counts against the app's CPU budget (Streamlit Cloud throttles heavy apps).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Supply Chain Copilot",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ───────────────────────────────────────────────────────────
from src.forecasting.data_generator import generate_demand_dataset
from src.forecasting.pipeline import ForecastingPipeline
from src.forecasting.models import MovingAverage, ExponentialSmoothing, SARIMAModel
from src.optimization.constraints import WorkforceConstraints
from src.optimization.pipeline import OptimizationPipeline
from src.analysis.deviation import build_deviation_report
from src.analysis.sensitivity import SensitivityAnalysis
from src.interpretation.context_builder import build_context
from src.ingestion.schema_detector import detect_schema_rules, detect_schema_llm, ColumnMapping
from src.ingestion.data_loader import load_file, apply_mapping, usable_skus


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data():
    path = "data/demand_data.csv"
    if Path(path).exists():
        return pd.read_csv(path, parse_dates=["date"])
    df = generate_demand_dataset(n_weeks=156, n_skus=3)
    Path("data").mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    return df


@st.cache_data(show_spinner=False, max_entries=64)
def forecast_sku(df_sku, sku, test_weeks, use_sarima=False):
    """Fit and rank the forecasting models for one SKU (the slow step, ~7-10 s).

    Cached on (data, sku, test_weeks, use_sarima): moving a constraint slider or going back
    to an already-computed SKU does not refit the models. SARIMA is by far the most
    expensive model (~8 s and ~14 CPU-s per SKU), so it is opt-in (see SARIMA_ENABLED).
    """
    models = [MovingAverage(window=12), ExponentialSmoothing(seasonal_periods=52)]
    if use_sarima:
        models.append(SARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)))
    fc_pipe = ForecastingPipeline(test_weeks=test_weeks, deviation_threshold_pct=20.0, models=models)
    sku_col = "sku" if "sku" in df_sku.columns else None
    fc_all = fc_pipe.run(df_sku, sku_col=sku_col)
    valid = [r for r in fc_all.get(str(sku), []) if not np.isnan(r.mape)]
    if not valid:
        raise ValueError(
            f"No forecast could be produced for '{sku}': the series has too few weeks "
            f"(need more than {test_weeks}) or every model failed. "
            "Pick another SKU or reduce the test horizon."
        )
    return valid


AUTO_MODEL = "Auto (lowest MAPE)"


def pick_forecast(results, model_choice):
    """Returns the forecast of the chosen model, or the lowest-MAPE one in Auto mode."""
    if model_choice == AUTO_MODEL:
        return min(results, key=lambda r: r.mape)
    for r in results:
        if r.model_name.startswith(model_choice):
            return r
    raise ValueError(f"Model '{model_choice}' failed on this SKU. Pick another model.")


def run_pipeline(df, sku, test_weeks, constraints, use_sarima=False, model_choice=AUTO_MODEL):
    if "sku" in df.columns:
        df = df[df["sku"] == sku]  # only the selected SKU, not every series in the file
    fc_results = forecast_sku(df, sku, test_weeks, use_sarima)
    fc = pick_forecast(fc_results, model_choice)

    opt_pipe = OptimizationPipeline(constraints=constraints, integer=True)
    alloc = opt_pipe.run(fc.test_dates, fc.forecast, sku=sku)

    dev = build_deviation_report(fc, alloc)

    sa = SensitivityAnalysis(base_demand=fc.forecast, base_constraints=constraints)
    sensitivity = sa.run_all()

    context = build_context(fc, alloc, dev, sensitivity)
    return fc, fc_results, alloc, dev, sensitivity, context


# SARIMA is only offered when explicitly enabled (ENABLE_SARIMA=1 in .env). A public
# deployment leaves it off so a visitor cannot trigger the heavy computation.
SARIMA_ENABLED = os.environ.get("ENABLE_SARIMA", "").strip().lower() in ("1", "true", "yes")


def metric_card(col, label, value, delta=None, delta_color="normal"):
    col.metric(label, value, delta=delta, delta_color=delta_color)


# ══════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📦 Supply Chain Copilot")
    st.caption("AI-powered forecasting & workforce optimisation")
    st.divider()

    st.subheader("Dataset")
    data_source = st.radio("Source", ["Demo dataset", "Upload my file"], horizontal=True)

    if data_source == "Demo dataset":
        df_active = load_data().copy()
        sku_list = sorted(df_active["sku"].unique().tolist())
        sku = st.selectbox("SKU", sku_list, index=0)

    else:
        uploaded = st.file_uploader(
            "CSV or Excel file", type=["csv", "xlsx", "xls"],
            help="Any tabular file with at least a date column and a numeric demand column.",
        )

        if uploaded is None:
            st.info("Upload a file to continue.")
            st.stop()

        # Load raw file
        file_key = f"raw_{uploaded.name}_{uploaded.size}"
        if st.session_state.get("file_key") != file_key:
            raw_df = load_file(uploaded)
            st.session_state["raw_df"] = raw_df
            st.session_state["file_key"] = file_key
            # Auto-detect schema
            api_key_val = os.environ.get("ANTHROPIC_API_KEY", "")
            use_llm = bool(api_key_val and not api_key_val.startswith("sk-ant-your"))
            try:
                if use_llm:
                    mapping = detect_schema_llm(raw_df, api_key_val)
                else:
                    mapping = detect_schema_rules(raw_df)
            except Exception as e:
                st.error(f"Schema detection failed: {e}")
                st.stop()
            st.session_state["mapping"] = mapping

        raw_df  = st.session_state["raw_df"]
        mapping = st.session_state["mapping"]

        st.caption(f"{len(raw_df):,} rows · {len(raw_df.columns)} columns detected")

        # Show warnings from detector
        for w in mapping.warnings:
            st.warning(w)

        # ── Column mapping UI ────────────────────────────────────────
        st.markdown("**Column mapping** — review and correct if needed")
        all_cols = list(raw_df.columns)

        def conf_label(col):
            c = mapping.confidence.get(col, "low")
            return {"high": "✓", "medium": "~", "low": "?"}[c]

        date_col = st.selectbox(
            f"Date column {conf_label(mapping.date_col)}",
            all_cols, index=all_cols.index(mapping.date_col),
        )
        demand_col = st.selectbox(
            f"Demand column {conf_label(mapping.demand_col)}",
            all_cols, index=all_cols.index(mapping.demand_col),
        )
        sku_options = ["(none)"] + all_cols
        sku_default = sku_options.index(mapping.sku_col) if mapping.sku_col in sku_options else 0
        sku_col_sel = st.selectbox("SKU / group column (optional)", sku_options, index=sku_default)

        confirmed_mapping = ColumnMapping(
            date_col=date_col,
            demand_col=demand_col,
            sku_col=sku_col_sel if sku_col_sel != "(none)" else None,
        )

        map_key = f"{date_col}_{demand_col}_{sku_col_sel}"
        if st.session_state.get("map_key") != map_key:
            try:
                df_active = apply_mapping(raw_df, confirmed_mapping)
                st.session_state["df_active"] = df_active
                st.session_state["map_key"] = map_key
            except Exception as e:
                st.error(f"Could not apply mapping: {e}")
                st.stop()

        df_active = st.session_state["df_active"]

        # SKU selector (or single series)
        if "sku" in df_active.columns:
            sku_list, n_hidden = usable_skus(df_active)
            sku = st.selectbox("SKU / group (largest volume first)", sku_list, index=0)
            if n_hidden:
                st.caption(f"{n_hidden} SKU(s) hidden: too few active weeks to forecast.")
        else:
            df_active["sku"] = "Series"
            sku = "Series"

        with st.expander("Preview (first 5 rows)"):
            st.dataframe(df_active.head(), use_container_width=True, hide_index=True)

    test_weeks = st.slider("Test horizon (weeks)", 4, 26, 13)
    use_sarima = False
    if SARIMA_ENABLED:
        use_sarima = st.checkbox(
            "Include SARIMA model (slower)", value=False,
            help="Adds a third forecasting model. Accurate on strongly seasonal data, but it "
                 "takes several seconds per SKU and a lot of CPU.",
        )
    model_options = [AUTO_MODEL, "MovingAverage", "ExponentialSmoothing"] + (["SARIMA"] if use_sarima else [])
    model_choice = st.selectbox(
        "Forecast model", model_options,
        help="Auto keeps the model with the lowest MAPE on the test horizon. "
             "Pick one explicitly to compare it or to drive the optimisation with it.",
    )

    st.divider()
    st.subheader("Workforce Constraints")

    capacity = st.number_input("Capacity / worker / week (units)", 10, 500, 50, step=5)
    cost_pw = st.number_input("Cost / worker / week (EUR)", 100, 5000, 900, step=50)
    max_w = st.slider("Max workers available", 5, 50, 20)
    budget = st.number_input("Total budget (EUR)", 50_000, 1_000_000, 175_000, step=5_000)
    min_sl = st.slider("Min service level (%)", 70, 100, 92) / 100
    ramp = st.slider("Max ramp-up / ramp-down (workers/week)", 1, 10, 3)

    constraints = WorkforceConstraints(
        capacity_per_worker=float(capacity),
        cost_per_worker=float(cost_pw),
        max_workers=int(max_w),
        budget_total=float(budget),
        min_service_level=min_sl,
        unmet_demand_penalty=4_000.0,
        max_ramp_up=int(ramp),
        max_ramp_down=int(ramp),
        currency="EUR",
    )

    st.divider()
    run_btn = st.button("▶ Run pipeline", type="primary", use_container_width=True)

    st.divider()
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    llm_available = bool(api_key and not api_key.startswith("sk-ant-your"))
    if llm_available:
        st.success("LLM: connected")
    else:
        st.warning("LLM: set ANTHROPIC_API_KEY in .env to unlock AI Insights")


# ══════════════════════════════════════════════════════════════════════
#  Run pipeline (cached in session state)
# ══════════════════════════════════════════════════════════════════════

cache_key = f"{sku}_{test_weeks}_{use_sarima}_{model_choice}_{capacity}_{cost_pw}_{max_w}_{budget}_{min_sl}_{ramp}_{st.session_state.get('map_key','demo')}"

if run_btn or "results" not in st.session_state or st.session_state.get("cache_key") != cache_key:
    with st.spinner("Running pipeline…"):
        try:
            fc, fc_results, alloc, dev, sensitivity, context = run_pipeline(
                df_active, sku, test_weeks, constraints, use_sarima, model_choice
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()
    st.session_state["results"] = (fc, fc_results, alloc, dev, sensitivity, context)
    st.session_state["cache_key"] = cache_key

fc, fc_results, alloc, dev, sensitivity, context = st.session_state["results"]

# ══════════════════════════════════════════════════════════════════════
#  Tabs
# ══════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Forecast", "👷 Optimisation", "🔍 Analysis", "🤖 AI Insights"]
)

# ─────────────────────────────────────────────────────────────────────
# TAB 1 — FORECAST
# ─────────────────────────────────────────────────────────────────────
with tab1:
    st.header(f"Demand Forecast — {sku}")

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Model", fc.model_name.split("(")[0])
    metric_card(c2, "MAPE", f"{fc.mape:.2f}%")
    metric_card(c3, "MAE", f"{fc.mae:.1f} units")
    metric_card(c4, "Significant deviations", str(len(fc.deviations)))
    st.caption("MAPE by model on the test horizon: " + " · ".join(
        f"{r.model_name.split('(')[0]} {r.mape:.2f}%" for r in sorted(fc_results, key=lambda r: r.mape)
    ))
    if "no seasonality" in fc.model_name:
        st.warning(
            "Yearly seasonality is switched off for this model: it needs at least two full years "
            "of training history (104 weeks after the test horizon). The forecast follows the "
            "trend only and will miss seasonal peaks."
        )

    st.divider()

    # Full history + forecast
    history = (
        df_active[df_active["sku"] == sku]
        .set_index("date")["demand"]
        .sort_index()
        .iloc[:-test_weeks]
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history.index, y=history.values,
        name="Historical", line=dict(color="#90a4ae", width=1.5), opacity=0.7,
    ))
    fig.add_vline(x=str(fc.train_end), line_dash="dash", line_color="#546e7a",
                  annotation_text="Train / Test split", annotation_position="top right")
    fig.add_trace(go.Scatter(
        x=fc.test_dates, y=fc.actual,
        name="Actual", line=dict(color="#1565c0", width=2.5),
        mode="lines+markers", marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=fc.test_dates, y=fc.forecast,
        name="Forecast", line=dict(color="#e53935", width=2, dash="dash"),
        mode="lines+markers", marker=dict(symbol="x", size=6),
    ))
    if not fc.deviations.empty:
        fig.add_trace(go.Scatter(
            x=fc.deviations["date"], y=fc.deviations["actual"],
            name="Deviation > 20%", mode="markers",
            marker=dict(color="orange", size=10, symbol="circle-open", line=dict(width=2)),
        ))
    fig.update_layout(
        title=dict(text="Demand history and forecast vs actual on the test period", y=0.97),
        margin=dict(t=90),
        height=430, legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Date", yaxis_title="Demand (units)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Error panel
    pct_err = np.where(
        fc.actual != 0,
        (fc.forecast - fc.actual) / fc.actual * 100,
        np.nan,
    )
    # Scale the y-axis to the actual errors (keeping the ±20% threshold just in view)
    # so that small errors stay readable instead of being crushed by a fixed ±50% range.
    max_abs_err = float(np.nanmax(np.abs(pct_err))) if np.isfinite(pct_err).any() else 0.0
    err_lim = max(max_abs_err * 1.25, 22)
    fig2 = go.Figure()
    fig2.add_hrect(y0=-20, y1=20, fillcolor="#43a047", opacity=0.07, line_width=0)
    fig2.add_bar(
        x=fc.test_dates, y=pct_err,
        marker_color=["#e53935" if abs(e) > 20 else "#1565c0" for e in pct_err],
        text=[f"{e:+.1f}%" if np.isfinite(e) else "" for e in pct_err],
        textposition="outside", cliponaxis=False,
        name="% Error",
    )
    fig2.add_hline(y=20, line_dash="dot", line_color="orange",
                   annotation_text="+20% alert", annotation_position="top left")
    fig2.add_hline(y=-20, line_dash="dot", line_color="orange",
                   annotation_text="-20% alert", annotation_position="bottom left")
    fig2.add_hline(y=0, line_color="black", line_width=1)
    fig2.update_layout(
        title="Weekly forecast error (above 0 = over-forecast, below 0 = under-forecast)",
        height=340, yaxis=dict(title="Forecast error (%)", range=[-err_lim, err_lim], ticksuffix="%"),
        showlegend=False, hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)

    if not fc.deviations.empty:
        with st.expander("Significant deviations (>20%)"):
            st.dataframe(
                fc.deviations[["date", "actual", "forecast", "pct_error"]]
                .rename(columns={"pct_error": "error (%)"})
                .assign(**{"error (%)": lambda d: d["error (%)"].round(1)}),
                use_container_width=True, hide_index=True,
            )


# ─────────────────────────────────────────────────────────────────────
# TAB 2 — OPTIMISATION
# ─────────────────────────────────────────────────────────────────────
with tab2:
    st.header(f"Workforce Optimisation — {sku}")

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Solver", alloc.solver_status)
    metric_card(c2, "Service level", f"{alloc.service_level:.1%}",
                delta=f"target {min_sl:.0%}", delta_color="off")
    metric_card(c3, "Total cost",
                f"{alloc.total_cost:,.0f} EUR",
                delta=f"{alloc.total_cost - alloc.baseline_flat_cost:+,.0f} vs flat",
                delta_color="inverse")
    metric_card(c4, "Avg workers / week", f"{alloc.workers.mean():.1f}")

    st.divider()

    # Allocation vs demand
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.1,
                        subplot_titles=("Forecasted demand vs demand covered by the plan",
                                        "Workers per week: optimised plan vs baselines"))

    fig.add_trace(go.Scatter(
        x=alloc.periods, y=alloc.demand,
        name="Forecasted demand", fill="tozeroy",
        line=dict(color="#1565c0", width=1.5), fillcolor="rgba(21,101,192,0.1)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=alloc.periods, y=alloc.covered_demand,
        name="Covered demand", line=dict(color="#2e7d32", width=2.5),
        mode="lines+markers", marker=dict(size=5),
    ), row=1, col=1)
    if alloc.unmet_demand.sum() > 0:
        fig.add_trace(go.Scatter(
            x=alloc.periods,
            y=alloc.covered_demand + alloc.unmet_demand,
            name="Unmet demand", fill="tonexty",
            line=dict(color="#e53935"), fillcolor="rgba(229,57,53,0.25)",
        ), row=1, col=1)

    width_ms = 3 * 24 * 3600 * 1000
    fig.add_trace(go.Bar(
        x=alloc.periods - pd.Timedelta(days=3),
        y=alloc.baseline_flat_workers,
        name="Flat baseline", marker_color="#90a4ae", opacity=0.6, width=width_ms,
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=alloc.periods,
        y=alloc.baseline_reactive_workers,
        name="Reactive baseline", marker_color="#ffb74d", opacity=0.7, width=width_ms,
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=alloc.periods + pd.Timedelta(days=3),
        y=alloc.workers,
        name="Optimised", marker_color="#1565c0", opacity=0.9, width=width_ms,
    ), row=2, col=1)
    fig.add_hline(y=max_w, line_dash="dash", line_color="red",
                  annotation_text=f"Max workers ({max_w})", row=2, col=1)

    fig.update_layout(
        height=560, barmode="overlay", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.06),
        margin=dict(t=90),
    )
    fig.update_yaxes(title_text="Units", row=1, col=1)
    fig.update_yaxes(title_text="Workers", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Cost comparison bar
    cost_labels = ["Optimised", "Flat baseline", "Reactive baseline"]
    cost_values = [alloc.total_cost, alloc.baseline_flat_cost, alloc.baseline_reactive_cost]
    fig3 = go.Figure(go.Bar(
        x=cost_labels, y=[v / 1000 for v in cost_values],
        marker_color=["#1565c0", "#90a4ae", "#ffb74d"],
        text=[f"{v/1000:.0f}k EUR" for v in cost_values],
        textposition="inside", insidetextanchor="end", textfont=dict(color="white", size=14),
    ))
    fig3.add_hline(y=budget / 1000, line_dash="dashdot", line_color="red",
                   annotation_text=f"Budget ({budget/1000:.0f}k EUR)")
    fig3.update_layout(
        title="Total cost over the horizon: optimised plan vs baselines",
        height=320, yaxis_title="Total cost (k EUR)",
        # Include the budget in the range, otherwise its line falls outside the chart
        showlegend=False, yaxis=dict(range=[0, max(max(cost_values), budget) / 1000 * 1.2]),
    )
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Weekly allocation table"):
        st.dataframe(alloc.to_dataframe().round(1), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────
# TAB 3 — ANALYSIS
# ─────────────────────────────────────────────────────────────────────
with tab3:
    st.header(f"Deviation & Sensitivity — {sku}")

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Forecast MAPE", f"{dev.forecast_mape:.2f}%")
    metric_card(c2, "Plan service level", f"{dev.plan_service_level:.1%}")
    metric_card(c3, "At-risk weeks", str(dev.weeks_at_risk))
    metric_card(c4, "Max coverage gap", f"{dev.max_coverage_gap:.0f} units")

    st.divider()
    st.subheader("Deviation report: plan vs actual demand")

    df_dev = dev.df
    dates_dev = pd.to_datetime(df_dev["date"])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.1,
                        subplot_titles=("Actual demand vs plan capacity (red area = capacity shortfall)",
                                        "Weekly forecast error (red bar = week where the plan fell short)"))

    fig.add_trace(go.Scatter(
        x=dates_dev, y=df_dev["actual"], name="Actual",
        line=dict(color="#1565c0", width=2.5), mode="lines+markers",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates_dev, y=df_dev["forecast"], name="Forecast",
        line=dict(color="#e53935", width=2, dash="dash"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates_dev, y=df_dev["plan_capacity"], name="Plan capacity",
        line=dict(color="#2e7d32", width=2), mode="lines",
    ), row=1, col=1)

    # One red block per short week, from plan capacity up to actual demand. A filled area
    # between the gap points would join non-adjacent weeks and paint gaps that do not exist.
    gap_mask = df_dev["plan_gap"] > 0
    if gap_mask.any():
        fig.add_trace(go.Bar(
            x=dates_dev[gap_mask], y=df_dev["plan_gap"][gap_mask],
            base=df_dev["plan_capacity"][gap_mask],
            width=3 * 24 * 3600 * 1000, marker_color="rgba(229,57,53,0.35)",
            marker_line=dict(color="#e53935", width=1), name="Coverage gap",
            hovertemplate="Short by %{y:.0f} units<extra></extra>",
        ), row=1, col=1)

    # Same scaling as the Forecast tab error chart: fit the actual errors, keep ±20% in view.
    dev_err = df_dev["forecast_error_pct"].to_numpy(dtype=float)
    dev_max_abs = float(np.nanmax(np.abs(dev_err))) if np.isfinite(dev_err).any() else 0.0
    dev_lim = max(dev_max_abs * 1.25, 22)
    fig.add_hrect(y0=-20, y1=20, fillcolor="#43a047", opacity=0.07, line_width=0, row=2, col=1)
    bar_colors = ["#e53935" if g > 0 else "#1565c0" for g in df_dev["plan_gap"]]
    fig.add_trace(go.Bar(
        x=dates_dev, y=dev_err,
        name="Forecast error (%)", marker_color=bar_colors, opacity=0.75,
        text=[f"{e:+.1f}%" if np.isfinite(e) else "" for e in dev_err],
        textposition="outside", cliponaxis=False,
    ), row=2, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="orange", row=2, col=1,
                  annotation_text="+20% alert", annotation_position="top left")
    fig.add_hline(y=-20, line_dash="dot", line_color="orange", row=2, col=1,
                  annotation_text="-20% alert", annotation_position="bottom left")
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)

    fig.update_layout(
        height=600, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.06),
        margin=dict(t=90),
    )
    fig.update_yaxes(title_text="Units", row=1, col=1)
    fig.update_yaxes(title_text="Forecast error (%)", row=2, col=1,
                     range=[-dev_lim, dev_lim], ticksuffix="%")
    # Bars make Plotly pad the date axis well past the last week; pin it to the test period
    fig.update_xaxes(range=[dates_dev.min() - pd.Timedelta(days=5), dates_dev.max() + pd.Timedelta(days=5)])
    st.plotly_chart(fig, use_container_width=True)

    # Sensitivity
    st.subheader("Sensitivity analysis")
    sens_col = st.columns(3)
    axis_configs = [
        ("demand",    "Demand shock",     "delta",  lambda x: f"{x:+.0%}",  "Demand change"),
        ("budget",    "Budget factor",    "delta",  lambda x: f"{x:.0%}",   "Budget (% of base)"),
        ("headcount", "Headcount loss",   "delta",  lambda x: f"-{x:.0f}w", "Workers removed"),
    ]
    for col, (key, title, xcol, xfmt, xlabel) in zip(sens_col, axis_configs):
        tdf = sensitivity[key]
        fig_s = make_subplots(specs=[[{"secondary_y": True}]])
        fig_s.add_trace(go.Scatter(
            x=tdf[xcol], y=tdf["total_cost"] / 1000,
            name="Cost (kEUR)", line=dict(color="#1565c0", width=2),
            mode="lines+markers",
        ), secondary_y=False)
        fig_s.add_trace(go.Scatter(
            x=tdf[xcol], y=tdf["service_level"] * 100,
            name="Service level (%)", line=dict(color="#e53935", width=2, dash="dash"),
            mode="lines+markers",
        ), secondary_y=True)
        fig_s.add_hline(y=min_sl * 100, line_dash="dot", line_color="#e53935", secondary_y=True)
        # Placed by hand: add_hline's own annotation uses the cost axis, not the secondary one
        fig_s.add_annotation(x=0, xref="paper", y=min_sl * 100, yref="y2",
                             text=f"target {min_sl:.0%}", showarrow=False,
                             xanchor="left", yanchor="bottom", font=dict(color="#e53935", size=11))
        infeasible = tdf[~tdf["feasible"]]
        if not infeasible.empty:
            for _, row in infeasible.iterrows():
                fig_s.add_vline(x=row[xcol], line_dash="dot", line_color="red", opacity=0.4)
        fig_s.update_layout(
            title=title, height=300, showlegend=False,
            xaxis=dict(
                tickvals=tdf[xcol].tolist(),
                ticktext=[xfmt(v) for v in tdf[xcol]],
                tickangle=-35,
            ),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        # Cost starts at zero so small cost moves are not exaggerated. The service-level axis
        # is zoomed on the band that matters (lowest value / target up to 100%) with explicit
        # ticks: left on auto, it was synced to the cost grid and got unreadable ticks.
        sl_pct = tdf["service_level"] * 100
        sl_low = max(0, int(min(sl_pct.min(), min_sl * 100) - 5) // 5 * 5)
        sl_step = 5 if 100 - sl_low <= 30 else 10
        fig_s.update_yaxes(title_text="Cost (kEUR)", secondary_y=False, title_font_color="#1565c0",
                           range=[0, tdf["total_cost"].max() / 1000 * 1.15])
        fig_s.update_yaxes(title_text="Service level", secondary_y=True, title_font_color="#e53935",
                           range=[sl_low, 102], tickmode="linear", tick0=sl_low, dtick=sl_step,
                           ticksuffix="%", showgrid=False)
        col.plotly_chart(fig_s, use_container_width=True)

    with st.expander("Raw sensitivity tables"):
        for key, tdf in sensitivity.items():
            st.caption(key.capitalize())
            st.dataframe(tdf.round(3), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────
# TAB 4 — AI INSIGHTS
# ─────────────────────────────────────────────────────────────────────
with tab4:
    st.header("AI Insights")

    if not llm_available:
        st.info(
            "Set **ANTHROPIC_API_KEY** in your `.env` file to unlock:\n\n"
            "- Executive summary (business-language narrative)\n"
            "- Interactive Q&A with what-if pipeline re-runs\n"
            "- Automatic edge-case scenario generation & robustness report"
        )
        with st.expander("Preview: context sent to the LLM"):
            st.code(context, language="text")

    else:
        from src.interpretation.interpreter import SupplyChainInterpreter
        from src.interpretation.scenarios import run_scenarios, to_dataframe
        from src.optimization.pipeline import OptimizationPipeline

        try:
            interpreter = SupplyChainInterpreter()
        except Exception as e:
            st.error(f"Could not initialise LLM: {e}")
            st.stop()

        # Executive summary
        st.subheader("Executive Summary")
        if st.button("Generate summary"):
            with st.spinner("Claude is writing the summary…"):
                summary = interpreter.executive_summary(context)
            st.session_state["summary"] = summary
        if "summary" in st.session_state:
            st.markdown(st.session_state["summary"])

        st.divider()

        # Q&A
        st.subheader("Ask a question")
        st.caption("Examples: 'What happens if demand rises 25%?' · 'Which weeks are at risk?' · 'What if budget drops 20%?'")
        question = st.text_input("Your question", key="qa_input")

        def make_rerun(base_demand, base_constraints):
            def rerun(axis, delta):
                d = base_demand.copy()
                c = base_constraints
                if axis == "demand":
                    d = d * (1 + delta)
                elif axis == "budget":
                    c = WorkforceConstraints(
                        capacity_per_worker=c.capacity_per_worker, cost_per_worker=c.cost_per_worker,
                        max_workers=c.max_workers, budget_total=c.budget_total * delta,
                        min_service_level=c.min_service_level, unmet_demand_penalty=c.unmet_demand_penalty,
                        max_ramp_up=c.max_ramp_up, max_ramp_down=c.max_ramp_down, currency=c.currency,
                    )
                elif axis == "headcount":
                    c = WorkforceConstraints(
                        capacity_per_worker=c.capacity_per_worker, cost_per_worker=c.cost_per_worker,
                        max_workers=max(1, c.max_workers - int(delta)), budget_total=c.budget_total,
                        min_service_level=c.min_service_level, unmet_demand_penalty=c.unmet_demand_penalty,
                        max_ramp_up=c.max_ramp_up, max_ramp_down=c.max_ramp_down, currency=c.currency,
                    )
                result = OptimizationPipeline(constraints=c).run(
                    pd.date_range(fc.test_dates[0], periods=len(d), freq="W-MON"), d, sku=sku,
                )
                return {"cost": result.total_cost, "service_level": result.service_level,
                        "avg_workers": float(result.workers.mean()), "status": result.solver_status}
            return rerun

        if st.button("Ask") and question:
            with st.spinner("Thinking…"):
                answer = interpreter.answer(
                    context, question,
                    rerun_fn=make_rerun(fc.forecast, constraints),
                )
            st.session_state.setdefault("qa_log", []).append(
                {"question": question, "answer": answer}
            )

        for entry in reversed(st.session_state.get("qa_log", [])):
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                st.write(entry["answer"])

        st.divider()

        # Robustness report
        st.subheader("Robustness Report")
        if st.button("Generate scenarios & run tests"):
            with st.spinner("Claude is generating scenarios…"):
                scenarios = interpreter.generate_scenarios(context, n=5)
            with st.spinner("Running scenarios through the LP…"):
                results = run_scenarios(scenarios, fc.forecast, constraints)
            st.session_state["scenario_results"] = results

        if "scenario_results" in st.session_state:
            results = st.session_state["scenario_results"]
            passed = sum(r.passed for r in results)
            st.metric("Scenarios passed", f"{passed} / {len(results)}")
            df_sc = to_dataframe(results)
            df_sc["passed"] = df_sc["passed"].map({True: "✓ PASS", False: "✗ FAIL"})
            df_sc["service_level"] = (df_sc["service_level"] * 100).round(1).astype(str) + "%"
            df_sc["total_cost"] = df_sc["total_cost"].map("{:,.0f} EUR".format)
            st.dataframe(
                df_sc[["scenario", "expected_risk", "description", "service_level", "total_cost", "passed"]],
                use_container_width=True, hide_index=True,
            )
