"""Professional Streamlit dashboard for historical openFDA recall data."""

from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    CLASSIFICATION_DIM_FILE,
    DATA_QUALITY_REPORT_FILE,
    EXTRACTION_METADATA_FILE,
    FIRM_SUMMARY_FILE,
    MONTHLY_SUMMARY_FILE,
    RECALLS_OUTPUT_FILE,
)
from src.transform import normalize_firm_name


REQUIRED_FILES = {
    "Recall records": RECALLS_OUTPUT_FILE,
    "Monthly summary": MONTHLY_SUMMARY_FILE,
    "Firm summary": FIRM_SUMMARY_FILE,
    "Classification reference": CLASSIFICATION_DIM_FILE,
    "Data-quality report": DATA_QUALITY_REPORT_FILE,
    "Extraction metadata": EXTRACTION_METADATA_FILE,
}

RECALL_COLUMNS = {
    "record_id", "source_record_hash", "recall_number", "event_id",
    "classification", "severity_label", "recalling_firm_raw",
    "recalling_firm_normalized", "product_description", "reason_for_recall",
    "recall_initiation_date", "report_date", "termination_date",
    "status_at_publication", "reporting_lag_days", "has_termination_date",
    "is_exact_duplicate",
}
MONTHLY_COLUMNS = {
    "report_month", "total_recalls", "three_month_rolling_avg_recalls"
}
FIRM_COLUMNS = {
    "recalling_firm_normalized", "recall_count", "unique_event_count",
    "class_i_count", "latest_report_date",
}
CLASSIFICATION_COLUMNS = {
    "classification", "severity_rank", "severity_label",
    "classification_description", "display_order",
}
DATE_COLUMNS = ["recall_initiation_date", "report_date", "termination_date"]
BOOLEAN_COLUMNS = ["has_termination_date", "is_exact_duplicate"]
NUMERIC_COLUMNS = ["severity_rank", "reporting_lag_days"]

NAVY = "#081525"
SLATE = "#122238"
TEAL = "#27C2B4"
BLUE = "#75A7F7"
TEXT = "#E8F0F8"
MUTED = "#A6B6C8"
BORDER = "#263A52"
AMBER = "#F0B45A"

PLOTLY_CONFIG = {
    "displayModeBar": "hover",
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
        "toggleSpikelines",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
        "fullscreen",
    ],
}


@st.cache_data(show_spinner=False)
def read_csv_cached(path_text: str, modified_time_ns: int) -> pd.DataFrame:
    """Read a generated CSV and invalidate the cache when it changes."""

    frame = pd.read_csv(Path(path_text), keep_default_na=False)
    return frame.replace("", pd.NA)


@st.cache_data(show_spinner=False)
def read_json_cached(path_text: str, modified_time_ns: int) -> dict[str, Any]:
    """Read a generated JSON object and invalidate the cache when it changes."""

    with Path(path_text).open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path_text}.")

    return payload


def load_csv(path: Path) -> pd.DataFrame:
    """Load one generated CSV using its modification time as a cache key."""

    return read_csv_cached(str(path), path.stat().st_mtime_ns)


def load_json(path: Path) -> dict[str, Any]:
    """Load one generated JSON document using a cache-aware reader."""

    return read_json_cached(str(path), path.stat().st_mtime_ns)


def convert_boolean_series(series: pd.Series) -> pd.Series:
    """Convert common CSV boolean representations into reliable booleans."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = series.astype("string").str.strip().str.lower()
    converted = normalized.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "false": False,
            "0": False,
            "no": False,
        }
    )
    return converted.fillna(False).astype(bool)


def prepare_recall_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply display-safe types to the processed recall table."""

    prepared = frame.copy()

    for column in DATE_COLUMNS:
        prepared[column] = pd.to_datetime(prepared[column], errors="coerce")

    for column in BOOLEAN_COLUMNS:
        prepared[column] = convert_boolean_series(prepared[column])

    for column in NUMERIC_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    return prepared


def build_business_base(frame: pd.DataFrame) -> pd.DataFrame:
    """Exclude exact source duplicates from business metrics, preserving facts."""

    return (
        frame.drop_duplicates(subset=["source_record_hash"], keep="first")
        .copy()
        .reset_index(drop=True)
    )


def normalize_firm_search(value: str) -> str:
    """Normalize a firm query with the same rules used by transformation."""

    normalized = normalize_firm_name(value)
    if pd.isna(normalized):
        return ""
    return str(normalized)


def filter_records(
    frame: pd.DataFrame,
    start_date: Any,
    end_date: Any,
    selected_classifications: list[str],
    selected_statuses: list[str],
    firm_query: str,
) -> pd.DataFrame:
    """Apply dashboard filters to the duplicate-safe business population."""

    filtered = frame.copy()
    start_timestamp = pd.Timestamp(start_date).normalize()
    end_timestamp = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)

    filtered = filtered[
        filtered["report_date"].ge(start_timestamp)
        & filtered["report_date"].lt(end_timestamp)
    ]

    if selected_classifications:
        filtered = filtered[
            filtered["classification"].isin(selected_classifications)
        ]

    if selected_statuses:
        filtered = filtered[
            filtered["status_at_publication"].isin(selected_statuses)
        ]

    normalized_query = normalize_firm_search(firm_query)
    if normalized_query:
        firm_values = (
            filtered["recalling_firm_normalized"].fillna("").astype(str)
        )
        filtered = filtered[
            firm_values.str.contains(
                normalized_query,
                case=False,
                regex=False,
            )
        ]

    return filtered.copy()


def build_monthly_trend(
    frame: pd.DataFrame,
    start_date: Any,
    end_date: Any,
) -> pd.DataFrame:
    """Build a continuous monthly trend and identify an incomplete final month."""

    start_period = pd.Period(start_date, freq="M")
    end_period = pd.Period(end_date, freq="M")
    full_months = pd.period_range(start_period, end_period, freq="M")

    if frame.empty:
        counts = pd.Series(0, index=full_months, dtype="int64")
    else:
        counts = (
            frame["report_date"]
            .dt.to_period("M")
            .value_counts()
            .reindex(full_months, fill_value=0)
            .sort_index()
        )

    trend = pd.DataFrame(
        {
            "report_month": full_months.astype(str),
            "month_start": full_months.to_timestamp(),
            "recall_records": counts.astype(int).to_numpy(),
        }
    )
    trend["three_month_average"] = (
        trend["recall_records"].rolling(3, min_periods=1).mean().round(2)
    )
    trend["is_partial_month"] = False
    trend["period_status"] = "Complete selected month"

    selected_end = pd.Timestamp(end_date).normalize()
    final_month_end = selected_end + pd.offsets.MonthEnd(0)
    if selected_end < final_month_end and not trend.empty:
        last_index = trend.index[-1]
        trend.loc[last_index, "is_partial_month"] = True
        trend.loc[last_index, "three_month_average"] = pd.NA
        trend.loc[last_index, "period_status"] = (
            f"Partial through {selected_end:%Y-%m-%d}"
        )

    return trend


def build_firm_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate firm metrics for the current filtered population."""

    columns = [
        "firm",
        "recall_records",
        "unique_events",
        "class_i_records",
        "class_i_share_pct",
        "latest_report_date",
        "median_reporting_lag_days",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    working = frame.copy()
    working["firm"] = (
        working["recalling_firm_normalized"]
        .fillna("UNKNOWN FIRM")
        .astype(str)
    )
    summary = (
        working.groupby("firm", dropna=False)
        .agg(
            recall_records=("record_id", "size"),
            unique_events=("event_id", lambda values: values.dropna().nunique()),
            class_i_records=(
                "classification",
                lambda values: values.eq("Class I").sum(),
            ),
            latest_report_date=("report_date", "max"),
            median_reporting_lag_days=("reporting_lag_days", "median"),
        )
        .reset_index()
    )
    summary["class_i_share_pct"] = (
        summary["class_i_records"]
        .div(summary["recall_records"])
        .mul(100)
        .round(1)
    )
    return (
        summary[columns]
        .sort_values(
            ["recall_records", "unique_events", "firm"],
            ascending=[False, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def build_classification_summary(
    frame: pd.DataFrame,
    dimension: pd.DataFrame,
) -> pd.DataFrame:
    """Combine filtered classification counts with the governed dimension."""

    counts = (
        frame["classification"]
        .fillna("Missing classification")
        .value_counts()
        .rename_axis("classification")
        .reset_index(name="recall_records")
    )
    summary = counts.merge(dimension, on="classification", how="left")
    summary["display_order"] = pd.to_numeric(
        summary["display_order"], errors="coerce"
    ).fillna(999)
    summary["severity_label"] = summary["severity_label"].fillna("Unmapped")
    summary["classification_description"] = summary[
        "classification_description"
    ].fillna("No classification definition is available.")
    total = int(summary["recall_records"].sum())
    summary["share_pct"] = (
        summary["recall_records"].div(total).mul(100).round(1)
        if total
        else 0.0
    )
    summary["display_label"] = (
        summary["classification"].astype(str)
        + " · "
        + summary["severity_label"].astype(str)
    )
    return summary.sort_values("display_order").reset_index(drop=True)


def format_status(value: Any) -> str:
    """Convert a machine-readable status into a human-readable label."""

    if value is None or pd.isna(value) or not str(value).strip():
        return "Not available"
    return str(value).strip().replace("_", " ").replace("-", " ").title()


def format_number(value: Any, decimals: int = 0) -> str:
    """Format a metric safely."""

    if value is None or pd.isna(value):
        return "Not available"
    if decimals == 0:
        return f"{int(round(float(value))):,}"
    return f"{float(value):,.{decimals}f}"


def format_days(value: Any) -> str:
    """Format a duration metric with a clear unit."""

    formatted = format_number(value)
    return formatted if formatted == "Not available" else f"{formatted} days"


def resolve_snapshot_timestamp(
    extraction_metadata: dict[str, Any],
    quality_report: dict[str, Any],
) -> str:
    """Resolve the best available timestamp for provenance display."""

    context = quality_report.get("pipeline_context", {})
    if not isinstance(context, dict):
        context = {}

    candidates = [
        extraction_metadata.get("snapshot_saved_at_utc"),
        extraction_metadata.get("extraction_finished_at_utc"),
        context.get("pipeline_completed_at_utc"),
    ]
    for value in candidates:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d %H:%M UTC")
    return "Timestamp unavailable"


def resolve_extraction_end_date(
    extraction_metadata: dict[str, Any],
    fallback: date,
) -> date:
    """Resolve the API query end date for responsible partial-month display."""

    parsed = pd.to_datetime(
        extraction_metadata.get("end_date"),
        format="%Y%m%d",
        errors="coerce",
    )
    return fallback if pd.isna(parsed) else parsed.date()


def validate_assets(
    recalls: pd.DataFrame,
    monthly: pd.DataFrame,
    firms: pd.DataFrame,
    classification: pd.DataFrame,
    quality_report: dict[str, Any],
    extraction_metadata: dict[str, Any],
) -> list[str]:
    """Return clear compatibility errors for required processed assets."""

    checks = [
        ("recalls_enriched.csv", recalls, RECALL_COLUMNS),
        ("monthly_summary.csv", monthly, MONTHLY_COLUMNS),
        ("firm_summary.csv", firms, FIRM_COLUMNS),
        ("dim_classification.csv", classification, CLASSIFICATION_COLUMNS),
    ]
    errors: list[str] = []
    for label, frame, required in checks:
        missing = sorted(required.difference(frame.columns))
        if missing:
            errors.append(f"{label} is missing columns: {', '.join(missing)}")

    if recalls.empty:
        errors.append("recalls_enriched.csv contains no records.")

    for key in ["overall_status", "summary", "issues", "reconciliation"]:
        if key not in quality_report:
            errors.append(f"data_quality_report.json is missing '{key}'.")

    for key in ["extraction_status", "records_extracted", "used_cached_data"]:
        if key not in extraction_metadata:
            errors.append(f"extraction_metadata.json is missing '{key}'.")

    return errors


def build_insights(
    frame: pd.DataFrame,
    firms: pd.DataFrame,
    trend: pd.DataFrame,
) -> dict[str, Any]:
    """Generate a structured, non-causal summary from filtered data."""

    if frame.empty:
        return {"primary": None, "supporting": []}

    classifications = frame["classification"].fillna("Missing classification")
    leader = classifications.value_counts().index[0]
    leader_count = int(classifications.eq(leader).sum())
    leader_share = leader_count / len(frame) * 100
    primary = {
        "label": "Leading classification",
        "value": f"{leader} · {leader_share:.1f}%",
        "detail": f"{leader_count:,} of {len(frame):,} filtered report records",
    }
    supporting: list[dict[str, str]] = []

    if not firms.empty:
        top_five_records = int(firms.head(5)["recall_records"].sum())
        concentration = top_five_records / len(frame) * 100
        supporting.append(
            {
                "label": "Top-five firm share",
                "value": f"{concentration:.1f}%",
                "detail": "Record concentration only—not company risk",
            }
        )

    complete_months = trend[~trend["is_partial_month"]]
    if len(complete_months) >= 6:
        recent = complete_months.tail(3)["recall_records"].mean()
        prior = complete_months.iloc[-6:-3]["recall_records"].mean()
        if prior > 0:
            change = (recent - prior) / prior * 100
            supporting.append(
                {
                    "label": "Recent 3-month change",
                    "value": f"{change:+.1f}%",
                    "detail": "Latest vs prior three complete months",
                }
            )

    return {"primary": primary, "supporting": supporting[:2]}


def short_quality_explanation(issue: dict[str, Any]) -> str:
    """Return a compact business explanation while preserving full detail elsewhere."""

    code = str(issue.get("code", "unknown"))
    known_explanations = {
        "missing_recall_number": (
            "Recall number is unavailable on these source records."
        ),
        "termination_date_contextual_missingness": (
            "Missing termination dates are contextual, not quality failures."
        ),
        "api_reported_count_mismatch": (
            "The API-reported total differs from the extracted row count."
        ),
        "exact_source_duplicates": (
            "Exact source duplicates are retained for auditability."
        ),
    }
    if code in known_explanations:
        return known_explanations[code]

    message = str(issue.get("message", "No explanation supplied.")).strip()
    first_sentence = message.split(".", 1)[0].strip()
    if len(first_sentence) > 105:
        first_sentence = first_sentence[:102].rstrip() + "…"
    return first_sentence + ("." if first_sentence and not first_sentence.endswith("…") else "")


def quality_finding_label(code: str) -> str:
    """Return a compact business label for a technical quality issue code."""

    known_labels = {
        "missing_recall_number": "Missing recall number",
        "termination_date_contextual_missingness": "Termination-date coverage",
        "api_reported_count_mismatch": "API count mismatch",
        "exact_source_duplicates": "Exact source duplicates",
    }
    return known_labels.get(code, format_status(code))


def render_css() -> None:
    """Apply a small, maintainable visual layer to native Streamlit."""

    st.markdown(
        f"""
        <style>
        .stApp {{ background: {NAVY}; color: {TEXT}; }}
        .block-container {{ max-width: 1500px; padding-top: .6rem;
            padding-bottom: 1rem; }}
        .app-header {{
            background: linear-gradient(120deg, #10243b 0%, #0b1a2d 72%);
            border: 1px solid {BORDER}; border-radius: 10px;
            padding: .62rem .85rem; margin-bottom: .35rem;
        }}
        .app-eyebrow {{ color: {TEAL}; font-size: .67rem; font-weight: 700;
            letter-spacing: .12em; text-transform: uppercase; }}
        .app-title {{ color: {TEXT}; font-size: 1.55rem; font-weight: 720;
            line-height: 1.1; margin: .12rem 0 .18rem; }}
        .app-subtitle {{ color: {MUTED}; font-size: .82rem;
            line-height: 1.3; }}
        .meta-row {{ display: flex; flex-wrap: wrap; gap: .3rem;
            margin-top: .38rem; }}
        .meta-chip {{ color: #C7D5E5; background: #132942;
            border: 1px solid #29445f; border-radius: 999px;
            padding: .14rem .45rem; font-size: .68rem; }}
        .responsible-note {{ border-left: 3px solid {TEAL}; color: #B8C7D7;
            background: #0e1d30; padding: .34rem .62rem; margin-bottom: .3rem;
            border-radius: 0 6px 6px 0; font-size: .75rem;
            line-height: 1.35; }}
        .section-intro {{ color: {MUTED}; font-size: .78rem;
            margin-top: -.35rem; margin-bottom: .2rem; }}
        .chart-title {{ color: {TEXT}; font-size: 1.02rem; font-weight: 650;
            line-height: 1.2; margin: .2rem 0 -.2rem; }}
        .context-strip {{ display: grid; grid-template-columns: repeat(3, 1fr);
            gap: .8rem; background: #0d1c2e; border: 1px solid {BORDER};
            border-radius: 8px; padding: .45rem .7rem; margin: .1rem 0 .35rem; }}
        .context-item + .context-item {{ border-left: 1px solid {BORDER};
            padding-left: .8rem; }}
        .context-label {{ color: {MUTED}; font-size: .66rem;
            letter-spacing: .04em; text-transform: uppercase; }}
        .context-value {{ color: {TEXT}; font-size: 1.1rem;
            font-weight: 650; line-height: 1.25; margin-top: .08rem; }}
        .insight-layout {{ display: grid; grid-template-columns: 1.45fr 1fr 1fr;
            gap: .55rem; margin-bottom: .2rem; }}
        .primary-insight, .supporting-stat {{ background: #0f2034;
            border: 1px solid {BORDER}; border-radius: 9px;
            padding: .62rem .75rem; min-height: 78px; }}
        .primary-insight {{ border-left: 3px solid {TEAL}; }}
        .insight-label {{ color: {MUTED}; font-size: .67rem;
            letter-spacing: .05em; text-transform: uppercase; }}
        .insight-value {{ color: {TEXT}; font-size: 1.05rem; font-weight: 680;
            line-height: 1.25; margin: .12rem 0; }}
        .insight-detail {{ color: #B9C8D8; font-size: .74rem;
            line-height: 1.3; }}
        .empty-state {{ border: 1px dashed #3A5068; border-radius: 12px;
            padding: 1.25rem; color: {MUTED}; background: #0d1b2d; }}
        div[data-testid="stSidebar"] {{ background: #0B1829; }}
        div[data-testid="stSidebarContent"] .block-container {{ padding-top: .75rem; }}
        div[data-testid="stSidebarContent"] h2 {{ font-size: 1.15rem !important;
            margin-bottom: -.25rem !important; }}
        div[data-testid="stSidebarContent"] div[data-testid="stVerticalBlock"] {{
            gap: .55rem; }}
        div[data-testid="stSidebar"] hr {{ margin: .35rem 0; }}
        div[data-testid="stSidebar"] label p {{ font-size: .8rem; }}
        div[data-testid="stMain"] h2 {{ font-size: 1.35rem !important;
            line-height: 1.2 !important; padding: .2rem 0 0 !important; }}
        div[data-testid="stMain"] h3 {{ font-size: 1.18rem !important;
            line-height: 1.2 !important; padding: .15rem 0 0 !important; }}
        div[data-testid="stMain"] h4 {{ font-size: 1rem !important;
            line-height: 1.2 !important; padding: .1rem 0 0 !important; }}
        div[data-testid="stMain"] div[data-testid="stMetric"] {{
            padding: .5rem .72rem; }}
        div[data-testid="stMain"] div[data-testid="stMetricValue"] {{
            font-size: 1.65rem; }}
        div[data-testid="stTabs"] div[data-baseweb="tab-list"] {{ gap: .15rem; }}
        div[data-testid="stTabs"] button[data-baseweb="tab"] {{
            height: 2.1rem; padding: 0 .62rem; }}
        div[data-testid="stTabs"] div[data-baseweb="tab-panel"] {{
            padding-top: .35rem; }}
        div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER};
            border-radius: 8px; overflow: hidden; }}
        @media (max-width: 900px) {{
            .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
            .app-title {{ font-size: 1.5rem; }}
            .insight-layout {{ grid-template-columns: 1fr; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(
    extraction_metadata: dict[str, Any],
    quality_report: dict[str, Any],
) -> None:
    """Render a compact product header and provenance summary."""

    snapshot = resolve_snapshot_timestamp(extraction_metadata, quality_report)
    extraction_status = format_status(
        extraction_metadata.get("extraction_status")
    )
    quality_status = format_status(quality_report.get("overall_status"))
    st.markdown(
        f"""
        <div class="app-header">
          <div class="app-eyebrow">Historical healthcare analytics</div>
          <div class="app-title">Drug Recall Intelligence</div>
          <div class="app-subtitle">
            Historical patterns in official openFDA drug enforcement reports.
          </div>
          <div class="meta-row">
            <span class="meta-chip">Snapshot &middot; {html.escape(snapshot)}</span>
            <span class="meta-chip">Extraction &middot; {html.escape(extraction_status)}</span>
            <span class="meta-chip">Quality &middot; {html.escape(quality_status)}</span>
          </div>
        </div>
        <div class="responsible-note">
          Historical analytics only&mdash;not a live alert, medical guidance or
          current-status source. &ldquo;Status at publication&rdquo; reflects the
          source record when published.
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_chart(
    figure: go.Figure,
    *,
    height: int,
    hovermode: str,
    uirevision: str,
) -> go.Figure:
    """Apply the dashboard theme and restrained interaction to Plotly charts."""

    figure.update_layout(
        height=height,
        margin={"l": 6, "r": 16, "t": 30, "b": 6},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": MUTED, "family": "Arial, sans-serif", "size": 11},
        hovermode=hovermode,
        hoverdistance=24,
        hoverlabel={
            "bgcolor": SLATE,
            "bordercolor": BORDER,
            "font": {"color": TEXT, "size": 11},
            "align": "left",
            "namelength": -1,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
            "font": {"color": MUTED, "size": 10},
        },
        dragmode="zoom",
        transition={"duration": 280, "easing": "cubic-in-out"},
        uirevision=uirevision,
    )
    figure.update_xaxes(
        color=MUTED,
        gridcolor="#20334A",
        linecolor="#3B5068",
        zerolinecolor="#3B5068",
        title_font={"color": MUTED, "size": 11},
        tickfont={"color": MUTED, "size": 10},
        automargin=True,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor="#536B86",
    )
    figure.update_yaxes(
        color=MUTED,
        gridcolor="#20334A",
        linecolor="#3B5068",
        zerolinecolor="#3B5068",
        title_font={"color": MUTED, "size": 11},
        tickfont={"color": MUTED, "size": 10},
        automargin=True,
    )
    return figure


def monthly_chart(trend: pd.DataFrame) -> go.Figure:
    """Build an interactive monthly records and rolling-average chart."""

    chart_frame = trend.copy()
    chart_frame["month_start"] = pd.to_datetime(chart_frame["month_start"])
    rolling_values = [
        None if pd.isna(value) else float(value)
        for value in chart_frame["three_month_average"]
    ]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=chart_frame["month_start"],
            y=chart_frame["recall_records"],
            mode="lines+markers",
            name="Recall records",
            line={"color": TEAL, "width": 2.5},
            marker={"color": TEAL, "size": 6, "line": {"color": NAVY, "width": 1}},
            customdata=chart_frame["period_status"],
            hovertemplate=(
                "<b>%{x|%B %Y}</b><br>"
                "Recall records: %{y:,}<br>"
                "Coverage: %{customdata}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart_frame["month_start"],
            y=rolling_values,
            mode="lines",
            name="3-month average",
            line={"color": BLUE, "width": 2, "dash": "dash"},
            connectgaps=False,
            hovertemplate=(
                "<b>%{x|%B %Y}</b><br>"
                "3-month average: %{y:.1f}<extra></extra>"
            ),
        )
    )

    partial = chart_frame[chart_frame["is_partial_month"]]
    if not partial.empty:
        figure.add_trace(
            go.Scatter(
                x=partial["month_start"],
                y=partial["recall_records"],
                mode="markers",
                name="Partial month",
                marker={
                    "color": AMBER,
                    "size": 11,
                    "symbol": "diamond",
                    "line": {"color": NAVY, "width": 1},
                },
                customdata=partial["period_status"],
                hovertemplate=(
                    "<b>%{x|%B %Y}</b><br>"
                    "Records so far: %{y:,}<br>"
                    "Coverage: %{customdata}<extra></extra>"
                ),
            )
        )

    figure.update_xaxes(title="Report month", tickformat="%b %Y")
    figure.update_yaxes(title="Recall records", rangemode="tozero")
    return style_chart(
        figure,
        height=300,
        hovermode="x unified",
        uirevision="monthly-reporting-trend",
    )


def classification_chart(summary: pd.DataFrame) -> go.Figure:
    """Build an interactive classification mix with direct values."""

    order = summary["display_label"].tolist()
    customdata = summary[
        ["classification", "severity_label", "share_pct"]
    ].to_numpy()
    figure = go.Figure(
        go.Bar(
            x=summary["recall_records"],
            y=summary["display_label"],
            orientation="h",
            name="Recall records",
            marker={"color": TEAL, "line": {"color": "#5DE0D4", "width": 0.5}},
            text=[f"{value:,}" for value in summary["recall_records"]],
            textposition="outside",
            textfont={"color": TEXT, "size": 11},
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Severity: %{customdata[1]}<br>"
                "Records: %{x:,}<br>"
                "Share: %{customdata[2]:.1f}%<extra></extra>"
            ),
        )
    )
    maximum_records = float(summary["recall_records"].max()) if not summary.empty else 0
    right_edge = maximum_records * 1.2 if maximum_records else 1
    figure.update_xaxes(
        title="Recall records",
        range=[0, right_edge],
        rangemode="tozero",
        automargin=True,
    )
    figure.update_yaxes(
        title=None,
        categoryorder="array",
        categoryarray=list(reversed(order)),
        showgrid=False,
    )
    styled = style_chart(
        figure,
        height=250,
        hovermode="closest",
        uirevision="classification-mix",
    )
    styled.update_layout(margin={"l": 6, "r": 48, "t": 28, "b": 6})
    return styled


def render_empty_state() -> None:
    """Render a consistent no-results state."""

    st.markdown(
        """
        <div class="empty-state">
          No enforcement report records match the current filters. Adjust the
          date range, classification, publication status or firm search.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(
    records: pd.DataFrame,
    classification_dimension: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> None:
    """Render the executive overview for the filtered population."""

    st.subheader("Overview")
    st.markdown(
        '<div class="section-intro">Metrics describe the selected historical report-record population.</div>',
        unsafe_allow_html=True,
    )
    if records.empty:
        render_empty_state()
        return

    total_records = len(records)
    unique_events = int(records["event_id"].dropna().nunique())
    unique_firms = int(records["recalling_firm_normalized"].dropna().nunique())
    class_i_records = int(records["classification"].eq("Class I").sum())
    class_i_share = class_i_records / total_records * 100

    cards = st.columns(4)
    cards[0].metric(
        "Recall records", format_number(total_records), border=True,
        help="openFDA report rows; not necessarily unique incidents",
    )
    cards[1].metric(
        "Unique recall events", format_number(unique_events), border=True,
        help="Distinct non-missing openFDA event identifiers",
    )
    cards[2].metric(
        "Recalling firms", format_number(unique_firms), border=True,
        help="Conservatively normalized firm names",
    )
    cards[3].metric(
        "Class I records", format_number(class_i_records), border=True,
        help=f"{class_i_share:.1f}% of filtered report records",
    )

    median_lag = records["reporting_lag_days"].median()
    termination_coverage = records["has_termination_date"].mean() * 100
    st.markdown(
        f"""
        <div class="context-strip">
          <div class="context-item" title="Historical days from recall initiation to report publication.">
            <div class="context-label">Median reporting lag</div>
            <div class="context-value">{html.escape(format_days(median_lag))}</div>
          </div>
          <div class="context-item" title="Share with a valid termination date; missing dates may be contextual.">
            <div class="context-label">Termination-date coverage</div>
            <div class="context-value">{termination_coverage:.1f}%</div>
          </div>
          <div class="context-item">
            <div class="context-label">Selected report period</div>
            <div class="context-value">{start_date:%b %y} &ndash; {end_date:%b %y}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    trend = build_monthly_trend(records, start_date, end_date)
    classification = build_classification_summary(
        records, classification_dimension
    )
    trend_column, mix_column = st.columns([1.65, 1])
    with trend_column:
        st.markdown(
            '<div class="chart-title">Monthly reporting trend</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            monthly_chart(trend),
            width="stretch",
            config=PLOTLY_CONFIG,
        )
        if trend["is_partial_month"].any():
            status = trend.loc[trend["is_partial_month"], "period_status"].iloc[0]
            st.caption(
                f"Latest month is {status.lower()} and is excluded from the "
                "three-month rolling average."
            )
        else:
            st.caption("Missing months are retained as zero before averaging.")

    with mix_column:
        st.markdown(
            '<div class="chart-title">Classification mix</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            classification_chart(classification),
            width="stretch",
            config=PLOTLY_CONFIG,
        )
        st.caption("Severity labels and direct values supplement color.")

    st.markdown("#### What stands out")
    firm_summary = build_firm_summary(records)
    insights = build_insights(records, firm_summary, trend)
    primary = insights["primary"]
    supporting_html = "".join(
        f"""
        <div class="supporting-stat">
          <div class="insight-label">{html.escape(item['label'])}</div>
          <div class="insight-value">{html.escape(item['value'])}</div>
          <div class="insight-detail">{html.escape(item['detail'])}</div>
        </div>
        """
        for item in insights["supporting"]
    )
    st.markdown(
        f"""
        <div class="insight-layout">
          <div class="primary-insight">
            <div class="insight-label">{html.escape(primary['label'])}</div>
            <div class="insight-value">{html.escape(primary['value'])}</div>
            <div class="insight-detail">{html.escape(primary['detail'])}</div>
          </div>
          {supporting_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Classification definitions"):
        context = classification_dimension.sort_values("display_order")[
            ["classification", "severity_label", "classification_description"]
        ]
        st.dataframe(
            context,
            width="stretch",
            hide_index=True,
            column_config={
                "classification": st.column_config.TextColumn("Classification"),
                "severity_label": st.column_config.TextColumn("Severity label"),
                "classification_description": st.column_config.TextColumn(
                    "Established definition", width="large"
                ),
            },
        )


def render_firm_intelligence(records: pd.DataFrame) -> None:
    """Render firm ranking and optional focused detail."""

    st.subheader("Firm intelligence")
    st.markdown(
        '<div class="section-intro">Firm volume reflects report-record representation, not company risk or safety performance.</div>',
        unsafe_allow_html=True,
    )
    if records.empty:
        render_empty_state()
        return

    summary = build_firm_summary(records)
    st.markdown("#### Compact ranking")
    ranking_columns = [
        "firm",
        "recall_records",
        "unique_events",
        "class_i_records",
        "class_i_share_pct",
        "latest_report_date",
    ]
    st.dataframe(
        summary.loc[:, ranking_columns].head(25),
        width="stretch",
        hide_index=True,
        height=370,
        column_config={
            "firm": st.column_config.TextColumn("Firm", width="medium"),
            "recall_records": st.column_config.NumberColumn(
                "Records", format="%d", width="small"
            ),
            "unique_events": st.column_config.NumberColumn(
                "Events", format="%d", width="small"
            ),
            "class_i_records": st.column_config.NumberColumn(
                "Class I", format="%d", width="small"
            ),
            "class_i_share_pct": st.column_config.NumberColumn(
                "Class I %", format="%.1f%%", width="small"
            ),
            "latest_report_date": st.column_config.DateColumn(
                "Latest report", format="YYYY-MM-DD", width="small"
            ),
        },
    )

    selected_firm = st.selectbox(
        "Inspect a firm",
        options=summary["firm"].sort_values().tolist(),
        index=None,
        placeholder="Choose a firm for detail",
    )
    if not selected_firm:
        st.caption("Select a firm to view its report-record history and details.")
        return

    selected = records[
        records["recalling_firm_normalized"].fillna("UNKNOWN FIRM").eq(selected_firm)
    ].copy()
    selected_summary = summary[summary["firm"].eq(selected_firm)].iloc[0]
    with st.container(border=True):
        st.markdown(f"#### {selected_firm}")
        detail_metrics = st.columns(4)
        detail_metrics[0].metric(
            "Recall records", format_number(selected_summary["recall_records"])
        )
        detail_metrics[1].metric(
            "Unique events", format_number(selected_summary["unique_events"])
        )
        detail_metrics[2].metric(
            "Class I records", format_number(selected_summary["class_i_records"])
        )
        detail_metrics[3].metric(
            "Median reporting lag",
            format_days(selected_summary["median_reporting_lag_days"]),
        )

        firm_trend = build_monthly_trend(
            selected,
            selected["report_date"].min().date(),
            selected["report_date"].max().date(),
        )
        st.markdown(
            '<div class="chart-title">Monthly report history</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            monthly_chart(firm_trend),
            width="stretch",
            config=PLOTLY_CONFIG,
        )
        st.caption("Hover over truncated table cells to read their full text.")
        st.dataframe(
            selected[
                [
                    "recall_number",
                    "report_date",
                    "classification",
                    "product_description",
                    "reason_for_recall",
                    "status_at_publication",
                ]
            ].sort_values("report_date", ascending=False),
            width="stretch",
            hide_index=True,
            height=330,
            column_config=explorer_column_config(),
        )


def explorer_column_config() -> dict[str, Any]:
    """Return business-friendly configuration for recall detail tables."""

    return {
        "recall_number": st.column_config.TextColumn("Recall #", width="small"),
        "report_date": st.column_config.DateColumn(
            "Report date", format="YYYY-MM-DD", width="small"
        ),
        "classification": st.column_config.TextColumn("Class", width="small"),
        "recalling_firm_raw": st.column_config.TextColumn(
            "Firm", width="medium"
        ),
        "product_description": st.column_config.TextColumn(
            "Product", width="medium"
        ),
        "reason_for_recall": st.column_config.TextColumn(
            "Recall reason", width="medium"
        ),
        "status_at_publication": st.column_config.TextColumn(
            "Status when published", width="medium"
        ),
    }


def render_recall_explorer(records: pd.DataFrame) -> None:
    """Render searchable, business-readable recall details."""

    st.subheader("Recall explorer")
    st.markdown(
        '<div class="section-intro">Inspect historical source records without exposing technical pipeline fields.</div>',
        unsafe_allow_html=True,
    )
    if records.empty:
        render_empty_state()
        return

    controls = st.columns([3, 1])
    search_text = controls[0].text_input(
        "Search records",
        placeholder="Recall number, firm, product or reason",
    )
    row_limit = controls[1].selectbox(
        "Rows displayed",
        options=[25, 50, 100, 250, 500],
        index=2,
    )

    explorer = records.copy()
    if search_text.strip():
        searchable_columns = [
            "recall_number",
            "recalling_firm_raw",
            "product_description",
            "reason_for_recall",
        ]
        searchable = (
            explorer[searchable_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
        )
        explorer = explorer[
            searchable.str.contains(search_text.strip(), case=False, regex=False)
        ].copy()

    display_columns = [
        "recall_number",
        "report_date",
        "classification",
        "recalling_firm_raw",
        "product_description",
        "reason_for_recall",
        "status_at_publication",
    ]
    explorer = explorer.sort_values("report_date", ascending=False)
    st.caption(
        f"{len(explorer):,} matching historical report records · "
        "Hover over truncated cells to read their full text."
    )
    st.dataframe(
        explorer[display_columns].head(row_limit),
        width="stretch",
        hide_index=True,
        height=500,
        column_config=explorer_column_config(),
    )

    download = explorer[display_columns].rename(
        columns={
            "recall_number": "Recall number",
            "report_date": "Report date",
            "classification": "Classification",
            "recalling_firm_raw": "Recalling firm",
            "product_description": "Product description",
            "reason_for_recall": "Reason for recall",
            "status_at_publication": "Status at publication",
        }
    )
    st.download_button(
        "Download filtered records",
        data=download.to_csv(index=False).encode("utf-8"),
        file_name="filtered_historical_drug_recalls.csv",
        mime="text/csv",
    )


def render_data_quality(
    quality_report: dict[str, Any],
    extraction_metadata: dict[str, Any],
) -> None:
    """Render dataset quality and extraction provenance separately."""

    st.subheader("Data quality")
    st.markdown(
        '<div class="section-intro">Validation findings and source provenance for the current processed snapshot.</div>',
        unsafe_allow_html=True,
    )
    summary = quality_report.get("summary", {})
    overall_status = quality_report.get("overall_status")
    compact_quality_status = {
        "passed": "Passed",
        "passed_with_warnings": "Warnings",
        "failed": "Failed",
    }.get(str(overall_status).lower(), format_status(overall_status))
    metrics = st.columns(4)
    metrics[0].metric(
        "Dataset quality", compact_quality_status,
        border=True,
    )
    metrics[1].metric(
        "Errors", format_number(summary.get("error_count", 0)), border=True
    )
    metrics[2].metric(
        "Warnings", format_number(summary.get("warning_count", 0)), border=True
    )
    metrics[3].metric(
        "Information", format_number(summary.get("info_count", 0)), border=True
    )
    st.caption(f"Overall validation status: {format_status(overall_status)}.")

    st.markdown("#### Dataset findings")
    issues = quality_report.get("issues", [])
    if issues:
        issue_rows = []
        technical_rows = []
        for issue in issues:
            code = str(issue.get("code", "unknown"))
            issue_rows.append(
                {
                    "Finding": quality_finding_label(code),
                    "Level": format_status(issue.get("severity")),
                    "Affected records": issue.get("affected_records", 0),
                    "Short explanation": short_quality_explanation(issue),
                }
            )
            technical_rows.append(
                {
                    "Technical code": code,
                    "Level": format_status(issue.get("severity")),
                    "Affected records": issue.get("affected_records", 0),
                    "Full explanation": issue.get("message", ""),
                }
            )
        st.dataframe(
            pd.DataFrame(issue_rows),
            width="stretch",
            hide_index=True,
            height=min(235, 38 + len(issue_rows) * 36),
            column_config={
                "Finding": st.column_config.TextColumn("Finding", width="medium"),
                "Level": st.column_config.TextColumn("Level", width="small"),
                "Affected records": st.column_config.NumberColumn(
                    "Affected records", format="%d", width="small"
                ),
                "Short explanation": st.column_config.TextColumn(
                    "Short explanation", width="large"
                ),
            },
        )
        with st.expander("Technical finding details"):
            st.dataframe(
                pd.DataFrame(technical_rows),
                width="stretch",
                hide_index=True,
                column_config={
                    "Technical code": st.column_config.TextColumn(
                        "Technical code", width="medium"
                    ),
                    "Level": st.column_config.TextColumn("Level", width="small"),
                    "Affected records": st.column_config.NumberColumn(
                        "Affected records", format="%d", width="small"
                    ),
                    "Full explanation": st.column_config.TextColumn(
                        "Full explanation", width="large"
                    ),
                },
            )
    else:
        st.success("No data-quality findings were reported.")

    reconciliation = quality_report.get("reconciliation", {})
    st.markdown("#### Record reconciliation")
    reconciliation_rows = pd.DataFrame(
        [
            {
                "Measure": "Records reported by openFDA",
                "Value": format_number(
                    reconciliation.get("records_reported_by_api")
                ),
            },
            {
                "Measure": "Records extracted",
                "Value": format_number(reconciliation.get("records_extracted")),
            },
            {
                "Measure": "Records transformed",
                "Value": format_number(reconciliation.get("records_transformed")),
            },
            {
                "Measure": "API count matches extraction",
                "Value": format_status(
                    reconciliation.get("api_reported_count_matches_extraction")
                ),
            },
            {
                "Measure": "Extraction count matches transformation",
                "Value": format_status(
                    reconciliation.get("row_count_matches_extraction")
                ),
            },
        ]
    )
    st.dataframe(
        reconciliation_rows,
        width="stretch",
        hide_index=True,
        column_config={
            "Measure": st.column_config.TextColumn("Measure", width="large"),
            "Value": st.column_config.TextColumn("Value", width="medium"),
        },
    )

    st.markdown("#### Extraction provenance")
    with st.container(border=True):
        provenance = st.columns(3)
        provenance[0].metric(
            "Extraction status",
            format_status(extraction_metadata.get("extraction_status")),
        )
        provenance[1].metric(
            "Records extracted",
            format_number(extraction_metadata.get("records_extracted")),
        )
        provenance[2].metric(
            "Cached fallback",
            "Yes" if extraction_metadata.get("used_cached_data") else "No",
        )
        st.caption(
            f"Source: {extraction_metadata.get('source', 'openFDA')} · "
            f"Query: {extraction_metadata.get('search_query', 'Unavailable')}"
        )

    detail_one, detail_two = st.columns(2)
    with detail_one.expander("Complete quality report"):
        st.json(quality_report)
    with detail_two.expander("Complete extraction metadata"):
        st.json(extraction_metadata)


def display_startup_errors(errors: list[str]) -> None:
    """Display actionable startup errors without exposing a stack trace."""

    st.error("The dashboard cannot load the processed dataset safely.")
    for error in errors:
        st.write(f"- {error}")
    st.code("python etl.py", language="bash")
    st.stop()


def resolve_selected_dates(value: Any) -> tuple[date, date]:
    """Normalize Streamlit date-input output into a complete range."""

    if isinstance(value, (tuple, list)) and len(value) == 2:
        return value[0], value[1]
    if isinstance(value, date):
        return value, value
    raise ValueError("A valid report-date range is required.")


def main() -> None:
    """Render the complete historical analytics product."""

    st.set_page_config(
        page_title="Drug Recall Intelligence",
        page_icon="DR",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_css()

    missing_files = [
        f"{label}: {path.as_posix()}"
        for label, path in REQUIRED_FILES.items()
        if not path.is_file()
    ]
    if missing_files:
        display_startup_errors([f"Missing {item}" for item in missing_files])

    try:
        recalls_raw = load_csv(RECALLS_OUTPUT_FILE)
        monthly = load_csv(MONTHLY_SUMMARY_FILE)
        firms = load_csv(FIRM_SUMMARY_FILE)
        classification = load_csv(CLASSIFICATION_DIM_FILE)
        quality_report = load_json(DATA_QUALITY_REPORT_FILE)
        extraction_metadata = load_json(EXTRACTION_METADATA_FILE)
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        display_startup_errors([f"Unable to read generated outputs: {exc}"])

    validation_errors = validate_assets(
        recalls_raw,
        monthly,
        firms,
        classification,
        quality_report,
        extraction_metadata,
    )
    if validation_errors:
        display_startup_errors(validation_errors)

    recalls = prepare_recall_data(recalls_raw)
    business_records = build_business_base(recalls)
    valid_dates = business_records["report_date"].dropna()
    if valid_dates.empty:
        display_startup_errors(["No valid report dates are available."])

    minimum_date = valid_dates.min().date()
    latest_observed_date = valid_dates.max().date()
    extraction_end_date = resolve_extraction_end_date(
        extraction_metadata, latest_observed_date
    )
    maximum_date = max(latest_observed_date, extraction_end_date)
    classifications = (
        classification.sort_values("display_order")["classification"]
        .astype(str)
        .tolist()
    )
    statuses = sorted(
        business_records["status_at_publication"].dropna().astype(str).unique()
    )

    render_header(extraction_metadata, quality_report)

    def reset_filters() -> None:
        st.session_state["report_date_filter"] = (minimum_date, maximum_date)
        st.session_state["classification_filter"] = classifications
        st.session_state["status_filter"] = statuses
        st.session_state["firm_filter"] = ""

    with st.sidebar:
        st.header("Filters")
        st.caption("Applied consistently across every dashboard section.")
        st.button("Reset filters", on_click=reset_filters, width="stretch")
        selected_range = st.date_input(
            "Report date range",
            value=(minimum_date, maximum_date),
            min_value=minimum_date,
            max_value=maximum_date,
            key="report_date_filter",
        )
        selected_classifications = st.multiselect(
            "Classification",
            options=classifications,
            default=classifications,
            key="classification_filter",
            help="An empty selection includes every classification.",
        )
        selected_statuses = st.multiselect(
            "Status at publication",
            options=statuses,
            default=statuses,
            key="status_filter",
            help="Historical status recorded when the enforcement report was published.",
        )
        firm_query = st.text_input(
            "Firm name contains",
            placeholder="Example: Johnson & Johnson",
            key="firm_filter",
        )

    try:
        selected_start, selected_end = resolve_selected_dates(selected_range)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    filtered = filter_records(
        business_records,
        selected_start,
        selected_end,
        selected_classifications,
        selected_statuses,
        firm_query,
    )
    with st.sidebar:
        st.divider()
        st.metric("Matching records", format_number(len(filtered)))
        st.caption(
            "Exact source duplicates are retained in the audit table but "
            "count once in dashboard metrics."
        )

    overview_tab, firms_tab, explorer_tab, quality_tab = st.tabs(
        ["Overview", "Firm intelligence", "Recall explorer", "Data quality"]
    )
    with overview_tab:
        render_overview(
            filtered,
            classification,
            selected_start,
            selected_end,
        )
    with firms_tab:
        render_firm_intelligence(filtered)
    with explorer_tab:
        render_recall_explorer(filtered)
    with quality_tab:
        render_data_quality(quality_report, extraction_metadata)

    st.divider()
    st.caption(
        "Source: official openFDA Drug Enforcement Reports API · Historical "
        "analytics only · Built with Python, pandas, Streamlit and Plotly"
    )


if __name__ == "__main__":
    main()
