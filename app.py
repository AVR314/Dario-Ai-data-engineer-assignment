"""Business-facing Streamlit app for historical drug recall intelligence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config import (
    DATA_QUALITY_REPORT_FILE,
    EXTRACTION_METADATA_FILE,
    FIRM_SUMMARY_FILE,
    MONTHLY_SUMMARY_FILE,
    RECALLS_OUTPUT_FILE,
)


st.set_page_config(
    page_title="Drug Recall Intelligence",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


REQUIRED_FILES = [
    RECALLS_OUTPUT_FILE,
    MONTHLY_SUMMARY_FILE,
    FIRM_SUMMARY_FILE,
    DATA_QUALITY_REPORT_FILE,
    EXTRACTION_METADATA_FILE,
]

CLASSIFICATION_ORDER = [
    "Class I",
    "Class II",
    "Class III",
    "Not Yet Classified",
]

STRING_COLUMNS = [
    "record_id",
    "source_record_hash",
    "recall_number",
    "event_id",
    "classification",
    "severity_label",
    "recalling_firm_raw",
    "recalling_firm_normalized",
    "product_description",
    "reason_for_recall",
    "voluntary_mandated",
    "distribution_pattern",
    "city",
    "state",
    "country",
    "status_at_publication",
]

DATE_COLUMNS = [
    "recall_initiation_date",
    "report_date",
    "termination_date",
]

BOOLEAN_COLUMNS = [
    "has_termination_date",
    "is_exact_duplicate",
    "has_repeated_recall_number",
    "invalid_recall_initiation_date_flag",
    "invalid_report_date_flag",
    "invalid_termination_date_flag",
    "negative_reporting_lag_flag",
    "negative_termination_duration_flag",
]

NUMERIC_COLUMNS = [
    "severity_rank",
    "reporting_lag_days",
    "termination_days",
    "recall_number_occurrence_count",
]


@st.cache_data(show_spinner=False)
def read_csv_cached(
    path_text: str,
    modified_time_ns: int,
) -> pd.DataFrame:
    """Read a CSV and invalidate the cache when the file changes."""

    if modified_time_ns <= 0:
        raise ValueError(
            "CSV modification time must be a positive integer."
        )

    frame = pd.read_csv(
        Path(path_text),
        keep_default_na=False,
    )

    # Preserve literal values such as "N/A", while treating empty CSV cells
    # as missing values.
    return frame.replace(
        "",
        pd.NA,
    )


@st.cache_data(show_spinner=False)
def read_json_cached(
    path_text: str,
    modified_time_ns: int,
) -> dict[str, Any]:
    """Read a JSON document and invalidate the cache when it changes."""

    if modified_time_ns <= 0:
        raise ValueError(
            "JSON modification time must be a positive integer."
        )

    with Path(path_text).open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object in {path_text}."
        )

    return payload


def load_csv(path: Path) -> pd.DataFrame:
    """Load one generated CSV using its modification time as cache input."""

    return read_csv_cached(
        str(path),
        path.stat().st_mtime_ns,
    )


def load_json(path: Path) -> dict[str, Any]:
    """Load one generated JSON file using a cache-aware reader."""

    return read_json_cached(
        str(path),
        path.stat().st_mtime_ns,
    )


def convert_boolean_series(
    series: pd.Series,
) -> pd.Series:
    """Convert CSV boolean values into reliable Python booleans."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

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


def prepare_recall_data(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare processed recall records for filtering and display."""

    prepared = frame.copy()

    for column in STRING_COLUMNS:
        if column in prepared.columns:
            prepared[column] = prepared[column].astype("string")

    for column in DATE_COLUMNS:
        if column in prepared.columns:
            prepared[column] = pd.to_datetime(
                prepared[column],
                errors="coerce",
            )

    for column in BOOLEAN_COLUMNS:
        if column in prepared.columns:
            prepared[column] = convert_boolean_series(
                prepared[column]
            )

    for column in NUMERIC_COLUMNS:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(
                prepared[column],
                errors="coerce",
            )

    return prepared


def build_business_base(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the business metric base.

    Exact source duplicates remain in the processed fact table for quality
    review but count only once in business metrics.
    """

    if "source_record_hash" not in frame.columns:
        return frame.copy()

    return (
        frame
        .drop_duplicates(
            subset=["source_record_hash"],
            keep="first",
        )
        .copy()
    )


def resolve_quality_status(
    quality_report: dict[str, Any],
) -> str:
    """Read the overall quality status from supported report layouts."""

    summary = quality_report.get(
        "summary",
        {},
    )

    if not isinstance(summary, dict):
        summary = {}

    candidates = [
        quality_report.get("overall_status"),
        quality_report.get("status"),
        summary.get("overall_status"),
        summary.get("status"),
    ]

    for value in candidates:
        if value:
            return str(value).strip().lower()

    return "unknown"


def resolve_extract_timestamp(
    extraction_metadata: dict[str, Any],
    quality_report: dict[str, Any],
) -> str:
    """Find and format the most relevant snapshot timestamp."""

    pipeline_context = quality_report.get(
        "pipeline_context",
        {},
    )

    if not isinstance(pipeline_context, dict):
        pipeline_context = {}

    candidates = [
        extraction_metadata.get("extracted_at_utc"),
        extraction_metadata.get("extraction_completed_at_utc"),
        extraction_metadata.get("extraction_timestamp_utc"),
        extraction_metadata.get("snapshot_saved_at_utc"),
        pipeline_context.get("pipeline_completed_at_utc"),
    ]

    for value in candidates:
        if not value:
            continue

        parsed = pd.to_datetime(
            value,
            utc=True,
            errors="coerce",
        )

        if pd.notna(parsed):
            return parsed.strftime(
                "%Y-%m-%d %H:%M UTC"
            )

        return str(value)

    return "Timestamp unavailable"


def ordered_classifications(
    frame: pd.DataFrame,
) -> list[str]:
    """Return observed classifications in a stable business order."""

    observed = {
        str(value)
        for value in frame["classification"].dropna()
    }

    expected = [
        value
        for value in CLASSIFICATION_ORDER
        if value in observed
    ]

    unexpected = sorted(
        observed.difference(CLASSIFICATION_ORDER)
    )

    return expected + unexpected


def filter_records(
    frame: pd.DataFrame,
    start_date: Any,
    end_date: Any,
    selected_classifications: list[str],
    selected_statuses: list[str],
    firm_query: str,
) -> pd.DataFrame:
    """Apply all dashboard filters to the business fact table."""

    filtered = frame.copy()

    end_timestamp = (
        pd.Timestamp(end_date)
        + pd.Timedelta(days=1)
        - pd.Timedelta(microseconds=1)
    )

    filtered = filtered[
        filtered["report_date"].between(
            pd.Timestamp(start_date),
            end_timestamp,
            inclusive="both",
        )
    ]

    if selected_classifications:
        filtered = filtered[
            filtered["classification"].isin(
                selected_classifications
            )
        ]

    if selected_statuses:
        filtered = filtered[
            filtered["status_at_publication"].isin(
                selected_statuses
            )
        ]

    normalized_query = firm_query.strip()

    if normalized_query:
        firm_values = (
            filtered["recalling_firm_normalized"]
            .fillna("")
            .astype(str)
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
    """Create a continuous monthly trend for the selected date range."""

    full_months = pd.period_range(
        start=pd.Period(start_date, freq="M"),
        end=pd.Period(end_date, freq="M"),
        freq="M",
    )

    if frame.empty:
        counts = pd.Series(
            0,
            index=full_months,
            dtype="int64",
        )

    else:
        report_month = (
            frame["report_date"]
            .dt.to_period("M")
        )

        counts = (
            report_month
            .value_counts()
            .sort_index()
            .reindex(
                full_months,
                fill_value=0,
            )
        )

    trend = pd.DataFrame(
        {
            "Report month": full_months.astype(str),
            "Recall records": counts.astype(int).values,
        }
    )

    trend["3-month average"] = (
        trend["Recall records"]
        .rolling(
            window=3,
            min_periods=1,
        )
        .mean()
        .round(2)
    )

    return trend


def build_firm_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Create firm-level metrics for the currently filtered records."""

    output_columns = [
        "Firm",
        "Recall records",
        "Unique events",
        "Class I records",
        "Class I share",
        "Median reporting lag",
        "First report",
        "Latest report",
    ]

    if frame.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    working = frame.copy()

    working["Firm"] = (
        working["recalling_firm_normalized"]
        .fillna("UNKNOWN FIRM")
        .astype(str)
    )

    summary = (
        working
        .groupby(
            "Firm",
            dropna=False,
        )
        .agg(
            **{
                "Recall records": (
                    "record_id",
                    "size",
                ),
                "Unique events": (
                    "event_id",
                    lambda values: int(
                        values.dropna().nunique()
                    ),
                ),
                "Class I records": (
                    "classification",
                    lambda values: int(
                        values.eq("Class I").sum()
                    ),
                ),
                "Median reporting lag": (
                    "reporting_lag_days",
                    "median",
                ),
                "First report": (
                    "report_date",
                    "min",
                ),
                "Latest report": (
                    "report_date",
                    "max",
                ),
            }
        )
        .reset_index()
    )

    summary["Class I share"] = (
        summary["Class I records"]
        .div(summary["Recall records"])
        .mul(100)
        .round(1)
    )

    return (
        summary[output_columns]
        .sort_values(
            by=[
                "Recall records",
                "Firm",
            ],
            ascending=[
                False,
                True,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def format_metric_number(
    value: Any,
    decimals: int = 0,
) -> str:
    """Format a numeric KPI or return a readable missing value."""

    if value is None or pd.isna(value):
        return "Not available"

    if decimals == 0:
        return f"{int(round(float(value))):,}"

    return f"{float(value):,.{decimals}f}"


def format_days_metric(
    value: Any,
) -> str:
    """Format a duration metric without producing invalid text."""

    if value is None or pd.isna(value):
        return "Not available"

    return (
        format_metric_number(value)
        + " days"
    )


def display_missing_files(
    missing_files: list[Path],
) -> None:
    """Display instructions when generated ETL outputs are unavailable."""

    st.error(
        "The dashboard cannot start because required ETL outputs "
        "are missing."
    )

    st.code(
        "python etl.py",
        language="bash",
    )

    st.write(
        "Missing files:"
    )

    for path in missing_files:
        st.write(
            f"- `{path.as_posix()}`"
        )

    st.stop()


def main() -> None:
    """Render the complete Streamlit dashboard."""

    missing_files = [
        path
        for path in REQUIRED_FILES
        if not path.is_file()
    ]

    if missing_files:
        display_missing_files(
            missing_files
        )

    recalls_raw = load_csv(
        RECALLS_OUTPUT_FILE
    )

    stored_monthly_summary = load_csv(
        MONTHLY_SUMMARY_FILE
    )

    stored_firm_summary = load_csv(
        FIRM_SUMMARY_FILE
    )

    quality_report = load_json(
        DATA_QUALITY_REPORT_FILE
    )

    extraction_metadata = load_json(
        EXTRACTION_METADATA_FILE
    )

    recalls = prepare_recall_data(
        recalls_raw
    )

    business_records = build_business_base(
        recalls
    )

    valid_report_dates = (
        business_records["report_date"]
        .dropna()
    )

    if valid_report_dates.empty:
        st.error(
            "No valid report dates are available in the processed data."
        )
        st.stop()

    minimum_date = (
        valid_report_dates.min().date()
    )

    maximum_date = (
        valid_report_dates.max().date()
    )

    quality_status = resolve_quality_status(
        quality_report
    )

    extraction_status = str(
        extraction_metadata.get(
            "extraction_status",
            "unknown",
        )
    )

    used_cached_data = bool(
        extraction_metadata.get(
            "used_cached_data",
            False,
        )
    )

    extract_timestamp = resolve_extract_timestamp(
        extraction_metadata,
        quality_report,
    )

    st.title(
        "Drug Recall Intelligence Dashboard"
    )

    st.markdown(
        """
        Historical intelligence on FDA drug recall reports, reporting
        patterns, severity classifications, and recalling firms.
        """
    )

    st.caption(
        f"Snapshot: {extract_timestamp} | "
        f"Extraction status: {extraction_status} | "
        f"Quality status: {quality_status}"
    )

    st.info(
        "This product is an analytical and educational view of historical "
        "openFDA enforcement reports. It must not be used for medical "
        "decisions, public safety alerts, or tracking the current lifecycle "
        "of a recall."
    )

    if used_cached_data:
        st.warning(
            "The latest ETL run used the committed local raw snapshot "
            "because the live API was unavailable."
        )

    with st.sidebar:
        st.header(
            "Dashboard filters"
        )

        selected_date_range = st.date_input(
            "Report date range",
            value=(
                minimum_date,
                maximum_date,
            ),
            min_value=minimum_date,
            max_value=maximum_date,
        )

        if (
            isinstance(
                selected_date_range,
                (tuple, list),
            )
            and len(selected_date_range) == 2
        ):
            selected_start_date = (
                selected_date_range[0]
            )

            selected_end_date = (
                selected_date_range[1]
            )

        else:
            selected_start_date = (
                selected_date_range
            )

            selected_end_date = (
                selected_date_range
            )

        classification_options = (
            ordered_classifications(
                business_records
            )
        )

        selected_classifications = st.multiselect(
            "Classification",
            options=classification_options,
            default=classification_options,
            help=(
                "Leave the selection empty to include "
                "all classifications."
            ),
        )

        status_options = sorted(
            str(value)
            for value in business_records[
                "status_at_publication"
            ].dropna().unique()
        )

        selected_statuses = st.multiselect(
            "Status at publication",
            options=status_options,
            default=status_options,
            help=(
                "This is the status contained in the "
                "published enforcement record."
            ),
        )

        firm_query = st.text_input(
            "Firm name contains",
            placeholder="Example: Pfizer",
        )

        st.divider()

        st.caption(
            f"Prepared outputs: "
            f"{len(stored_monthly_summary):,} monthly rows and "
            f"{len(stored_firm_summary):,} firm rows."
        )

        st.caption(
            "Run `python etl.py` to refresh the local snapshot."
        )

    filtered_records = filter_records(
        business_records,
        start_date=selected_start_date,
        end_date=selected_end_date,
        selected_classifications=(
            selected_classifications
        ),
        selected_statuses=selected_statuses,
        firm_query=firm_query,
    )

    overview_tab, firms_tab, explorer_tab, quality_tab = (
        st.tabs(
            [
                "Overview",
                "Firm intelligence",
                "Recall explorer",
                "Data quality",
            ]
        )
    )

    with overview_tab:
        st.subheader(
            "Selected population"
        )

        if filtered_records.empty:
            st.warning(
                "No recall records match the current filters."
            )

        else:
            total_records = len(
                filtered_records
            )

            unique_events = (
                filtered_records["event_id"]
                .dropna()
                .nunique()
            )

            unique_firms = (
                filtered_records[
                    "recalling_firm_normalized"
                ]
                .dropna()
                .nunique()
            )

            class_i_records = int(
                filtered_records[
                    "classification"
                ]
                .eq("Class I")
                .sum()
            )

            median_reporting_lag = (
                filtered_records[
                    "reporting_lag_days"
                ]
                .median()
            )

            termination_coverage = (
                filtered_records[
                    "has_termination_date"
                ]
                .mean()
                * 100
            )

            metric_columns = st.columns(
                6
            )

            metric_columns[0].metric(
                "Recall records",
                format_metric_number(
                    total_records
                ),
                border=True,
            )

            metric_columns[1].metric(
                "Unique events",
                format_metric_number(
                    unique_events
                ),
                border=True,
            )

            metric_columns[2].metric(
                "Recalling firms",
                format_metric_number(
                    unique_firms
                ),
                border=True,
            )

            metric_columns[3].metric(
                "Class I records",
                format_metric_number(
                    class_i_records
                ),
                border=True,
            )

            metric_columns[4].metric(
                "Median reporting lag",
                format_days_metric(
                    median_reporting_lag
                ),
                border=True,
            )

            metric_columns[5].metric(
                "Termination-date coverage",
                (
                    format_metric_number(
                        termination_coverage,
                        decimals=1,
                    )
                    + "%"
                ),
                border=True,
            )

            monthly_trend = build_monthly_trend(
                filtered_records,
                start_date=selected_start_date,
                end_date=selected_end_date,
            )

            classification_counts = (
                filtered_records[
                    "classification"
                ]
                .fillna("Missing classification")
                .value_counts()
            )

            ordered_labels = [
                label
                for label in CLASSIFICATION_ORDER
                if label in classification_counts.index
            ]

            ordered_labels.extend(
                sorted(
                    label
                    for label in classification_counts.index
                    if label not in CLASSIFICATION_ORDER
                )
            )

            classification_chart = (
                classification_counts
                .reindex(ordered_labels)
                .rename_axis("Classification")
                .to_frame("Recall records")
            )

            trend_column, classification_column = (
                st.columns(
                    [
                        2,
                        1,
                    ]
                )
            )

            with trend_column:
                st.subheader(
                    "Monthly reporting trend"
                )

                st.line_chart(
                    monthly_trend.set_index(
                        "Report month"
                    ),
                    height=360,
                )

                st.caption(
                    "Missing calendar months are included with zero "
                    "records before the rolling average is calculated."
                )

            with classification_column:
                st.subheader(
                    "Classification mix"
                )

                st.bar_chart(
                    classification_chart,
                    height=360,
                )

            firm_summary = build_firm_summary(
                filtered_records
            )

            busiest_month = monthly_trend.loc[
                monthly_trend[
                    "Recall records"
                ].idxmax()
            ]

            top_classification = (
                classification_counts.idxmax()
            )

            top_firm = (
                firm_summary.iloc[0]
                if not firm_summary.empty
                else None
            )

            st.subheader(
                "Business observations"
            )

            insight_columns = st.columns(
                3
            )

            insight_columns[0].write(
                f"**Busiest report month:** "
                f"{busiest_month['Report month']} with "
                f"{int(busiest_month['Recall records']):,} records."
            )

            insight_columns[1].write(
                f"**Largest classification group:** "
                f"{top_classification} with "
                f"{int(classification_counts.max()):,} records."
            )

            if top_firm is not None:
                insight_columns[2].write(
                    f"**Most represented firm:** "
                    f"{top_firm['Firm']} with "
                    f"{int(top_firm['Recall records']):,} records."
                )

    with firms_tab:
        st.subheader(
            "Recalling-firm intelligence"
        )

        firm_summary = build_firm_summary(
            filtered_records
        )

        if firm_summary.empty:
            st.warning(
                "No firm-level results match the current filters."
            )

        else:
            st.dataframe(
                firm_summary.head(30),
                width="stretch",
                hide_index=True,
                column_config={
                    "Recall records": (
                        st.column_config.NumberColumn(
                            format="%d",
                        )
                    ),
                    "Unique events": (
                        st.column_config.NumberColumn(
                            format="%d",
                        )
                    ),
                    "Class I records": (
                        st.column_config.NumberColumn(
                            format="%d",
                        )
                    ),
                    "Class I share": (
                        st.column_config.NumberColumn(
                            format="%.1f%%",
                        )
                    ),
                    "Median reporting lag": (
                        st.column_config.NumberColumn(
                            format="%.1f",
                        )
                    ),
                    "First report": (
                        st.column_config.DateColumn(
                            format="YYYY-MM-DD",
                        )
                    ),
                    "Latest report": (
                        st.column_config.DateColumn(
                            format="YYYY-MM-DD",
                        )
                    ),
                },
            )

            selected_firm = st.selectbox(
                "Inspect a firm",
                options=(
                    firm_summary["Firm"]
                    .sort_values()
                    .tolist()
                ),
            )

            selected_firm_records = (
                filtered_records[
                    filtered_records[
                        "recalling_firm_normalized"
                    ]
                    .fillna("UNKNOWN FIRM")
                    .eq(selected_firm)
                ]
                .copy()
            )

            selected_firm_trend = (
                selected_firm_records
                .assign(
                    report_month=(
                        selected_firm_records[
                            "report_date"
                        ]
                        .dt.to_period("M")
                        .astype(str)
                    )
                )
                .groupby(
                    "report_month"
                )
                .size()
                .rename(
                    "Recall records"
                )
                .to_frame()
            )

            firm_metric_columns = st.columns(
                4
            )

            firm_metric_columns[0].metric(
                "Recall records",
                f"{len(selected_firm_records):,}",
                border=True,
            )

            firm_metric_columns[1].metric(
                "Unique events",
                (
                    f"{selected_firm_records['event_id'].dropna().nunique():,}"
                ),
                border=True,
            )

            firm_metric_columns[2].metric(
                "Class I records",
                (
                    f"{selected_firm_records['classification'].eq('Class I').sum():,}"
                ),
                border=True,
            )

            firm_metric_columns[3].metric(
                "Median reporting lag",
                format_days_metric(
                    selected_firm_records[
                        "reporting_lag_days"
                    ].median()
                ),
                border=True,
            )

            st.line_chart(
                selected_firm_trend,
                height=280,
            )

            firm_record_columns = [
                "recall_number",
                "event_id",
                "classification",
                "product_description",
                "reason_for_recall",
                "report_date",
                "status_at_publication",
            ]

            st.dataframe(
                selected_firm_records[
                    firm_record_columns
                ]
                .sort_values(
                    "report_date",
                    ascending=False,
                ),
                width="stretch",
                hide_index=True,
            )

    with explorer_tab:
        st.subheader(
            "Search and inspect recall records"
        )

        search_text = st.text_input(
            "Search recall number, event, firm, product, or reason",
            key="record_search",
        )

        explorer_records = (
            filtered_records.copy()
        )

        if search_text.strip():
            searchable_columns = [
                "recall_number",
                "event_id",
                "recalling_firm_raw",
                "product_description",
                "reason_for_recall",
            ]

            searchable_text = (
                explorer_records[
                    searchable_columns
                ]
                .fillna("")
                .astype(str)
                .agg(
                    " ".join,
                    axis=1,
                )
            )

            explorer_records = (
                explorer_records[
                    searchable_text.str.contains(
                        search_text.strip(),
                        case=False,
                        regex=False,
                    )
                ]
                .copy()
            )

        row_limit = st.select_slider(
            "Rows displayed",
            options=[
                25,
                50,
                100,
                250,
                500,
            ],
            value=100,
        )

        st.caption(
            f"{len(explorer_records):,} matching records."
        )

        display_columns = [
            "recall_number",
            "event_id",
            "classification",
            "recalling_firm_raw",
            "product_description",
            "reason_for_recall",
            "recall_initiation_date",
            "report_date",
            "termination_date",
            "reporting_lag_days",
            "status_at_publication",
        ]

        explorer_display = (
            explorer_records[
                display_columns
            ]
            .sort_values(
                "report_date",
                ascending=False,
            )
            .head(row_limit)
        )

        st.dataframe(
            explorer_display,
            width="stretch",
            hide_index=True,
            column_config={
                "recall_initiation_date": (
                    st.column_config.DateColumn(
                        format="YYYY-MM-DD",
                    )
                ),
                "report_date": (
                    st.column_config.DateColumn(
                        format="YYYY-MM-DD",
                    )
                ),
                "termination_date": (
                    st.column_config.DateColumn(
                        format="YYYY-MM-DD",
                    )
                ),
            },
        )

        download_data = (
            explorer_records[
                display_columns
            ]
            .sort_values(
                "report_date",
                ascending=False,
            )
            .to_csv(
                index=False,
            )
            .encode("utf-8")
        )

        st.download_button(
            label="Download filtered records as CSV",
            data=download_data,
            file_name="filtered_drug_recalls.csv",
            mime="text/csv",
        )

    with quality_tab:
        st.subheader(
            "Data quality and pipeline transparency"
        )

        quality_summary = quality_report.get(
            "summary",
            {},
        )

        if not isinstance(
            quality_summary,
            dict,
        ):
            quality_summary = {}

        quality_metric_columns = st.columns(
            4
        )

        quality_metric_columns[0].metric(
            "Overall status",
            quality_status,
            border=True,
        )

        quality_metric_columns[1].metric(
            "Errors",
            quality_summary.get(
                "error_count",
                0,
            ),
            border=True,
        )

        quality_metric_columns[2].metric(
            "Warnings",
            quality_summary.get(
                "warning_count",
                0,
            ),
            border=True,
        )

        quality_metric_columns[3].metric(
            "Information notices",
            quality_summary.get(
                "info_count",
                0,
            ),
            border=True,
        )

        issues = quality_report.get(
            "issues",
            [],
        )

        st.subheader(
            "Detected issues"
        )

        if issues:
            issues_frame = pd.DataFrame(
                issues
            )

            st.dataframe(
                issues_frame,
                width="stretch",
                hide_index=True,
            )

        else:
            st.success(
                "No data-quality issues were detected."
            )

        detail_column, metadata_column = (
            st.columns(
                2
            )
        )

        with detail_column:
            st.subheader(
                "Validation report"
            )

            with st.expander(
                "View complete quality report"
            ):
                st.json(
                    quality_report
                )

        with metadata_column:
            st.subheader(
                "Extraction metadata"
            )

            with st.expander(
                "View API extraction metadata"
            ):
                st.json(
                    extraction_metadata
                )

        st.subheader(
            "Validation approach"
        )

        st.markdown(
            """
            - Required schema and critical business fields are validated.
            - Missing values and placeholder values such as `N/A` are
              normalized and surfaced.
            - Exact duplicates are preserved for auditability but count only
              once in business metrics.
            - Invalid dates and negative historical durations are flagged.
            - Unexpected classification values remain visible for review.
            - Extracted and transformed record counts are reconciled.
            - Missing termination dates are reported separately because they
              may be expected or structural.
            """
        )

        st.subheader(
            "Known limitations"
        )

        st.markdown(
            """
            - The dashboard describes historical enforcement reports and is
              not a current safety-alert system.
            - The status field reflects the value published in the source
              record and must not be used to track the current recall
              lifecycle.
            - Reporting lag measures the historical difference between recall
              initiation and report publication. It does not measure the speed
              of every internal FDA or company process.
            - Firm-name normalization is intentionally conservative. It
              standardizes punctuation and casing without fuzzy matching that
              could merge different organizations.
            - Missing termination dates are not automatically evidence of bad
              source data.
            """
        )

        st.link_button(
            "Open official openFDA source",
            "https://open.fda.gov/apis/drug/enforcement/",
        )

    st.divider()

    st.caption(
        "Built with Python, pandas, Streamlit, and the openFDA Drug "
        "Enforcement Reports API."
    )


if __name__ == "__main__":
    main()