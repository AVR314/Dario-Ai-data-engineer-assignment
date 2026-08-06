"""Business analytics for transformed drug recall records."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.config import (
    END_DATE,
    ROLLING_WINDOW_MONTHS,
    START_DATE,
)


REQUIRED_ANALYTICS_COLUMNS = {
    "record_id",
    "source_record_hash",
    "event_id",
    "classification",
    "recalling_firm_normalized",
    "report_date",
    "reporting_lag_days",
    "has_termination_date",
}


class AnalyticsError(RuntimeError):
    """Raised when analytical outputs cannot be created safely."""


def _validate_frame(frame: pd.DataFrame) -> None:
    """Validate that the transformed dataset supports analytics."""

    if not isinstance(frame, pd.DataFrame):
        raise AnalyticsError(
            "The analytics layer requires a pandas DataFrame."
        )

    missing_columns = sorted(
        REQUIRED_ANALYTICS_COLUMNS.difference(frame.columns)
    )

    if missing_columns:
        raise AnalyticsError(
            "The transformed dataset is missing analytics columns: "
            + ", ".join(missing_columns)
        )


def _parse_month_bound(
    value: str,
    field_name: str,
) -> pd.Period:
    """Parse a YYYYMMDD boundary into a monthly period."""

    try:
        parsed = datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise AnalyticsError(
            f"{field_name} must use YYYYMMDD format. "
            f"Received: {value!r}"
        ) from exc

    return pd.Period(parsed, freq="M")


def _prepare_analytics_base(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the analytical base while preserving the source fact table.

    Exact duplicate source rows remain available in recalls_enriched.csv for
    quality review, but only one row per source_record_hash contributes to
    business metrics.
    """

    _validate_frame(frame)

    analytical = frame.copy()

    analytical = analytical.drop_duplicates(
        subset=["source_record_hash"],
        keep="first",
    )

    analytical["report_date"] = pd.to_datetime(
        analytical["report_date"],
        errors="coerce",
    )

    analytical["reporting_lag_days"] = pd.to_numeric(
        analytical["reporting_lag_days"],
        errors="coerce",
    )

    analytical["has_termination_date"] = (
        analytical["has_termination_date"]
        .fillna(False)
        .astype(bool)
    )

    return analytical


def build_monthly_summary(
    frame: pd.DataFrame,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """
    Build a continuous monthly recall summary.

    Missing months are inserted explicitly with zero recall counts before the
    rolling average is calculated. This prevents silent trend distortion.
    """

    start_month = _parse_month_bound(
        start_date,
        "start_date",
    )
    end_month = _parse_month_bound(
        end_date,
        "end_date",
    )

    if start_month > end_month:
        raise AnalyticsError(
            "start_date must not be later than end_date."
        )

    analytical = _prepare_analytics_base(frame)

    analytical["report_month_period"] = (
        analytical["report_date"].dt.to_period("M")
    )

    valid_month_records = analytical[
        analytical["report_month_period"].notna()
    ].copy()

    full_month_index = pd.period_range(
        start=start_month,
        end=end_month,
        freq="M",
    )

    if valid_month_records.empty:
        grouped = pd.DataFrame(index=full_month_index)

    else:
        grouped = valid_month_records.groupby(
            "report_month_period",
            observed=False,
        ).agg(
            total_recalls=("record_id", "size"),
            unique_firms=(
                "recalling_firm_normalized",
                lambda values: int(
                    values.dropna().nunique()
                ),
            ),
            median_reporting_lag_days=(
                "reporting_lag_days",
                "median",
            ),
            termination_date_coverage_pct=(
                "has_termination_date",
                lambda values: round(
                    float(values.mean()) * 100,
                    2,
                ),
            ),
        )

        classification_counts = (
            valid_month_records.groupby(
                [
                    "report_month_period",
                    "classification",
                ],
                observed=False,
            )
            .size()
            .unstack(fill_value=0)
        )

        classification_columns = {
            "Class I": "class_i_recalls",
            "Class II": "class_ii_recalls",
            "Class III": "class_iii_recalls",
            "Not Yet Classified": (
                "not_yet_classified_recalls"
            ),
        }

        for source_name, output_name in (
            classification_columns.items()
        ):
            if source_name in classification_counts.columns:
                grouped[output_name] = (
                    classification_counts[source_name]
                )
            else:
                grouped[output_name] = 0

    summary = grouped.reindex(full_month_index)

    count_columns = [
        "total_recalls",
        "unique_firms",
        "class_i_recalls",
        "class_ii_recalls",
        "class_iii_recalls",
        "not_yet_classified_recalls",
    ]

    for column in count_columns:
        if column not in summary.columns:
            summary[column] = 0

        summary[column] = (
            summary[column]
            .fillna(0)
            .astype("Int64")
        )

    if "median_reporting_lag_days" not in summary.columns:
        summary["median_reporting_lag_days"] = pd.NA

    if (
        "termination_date_coverage_pct"
        not in summary.columns
    ):
        summary["termination_date_coverage_pct"] = pd.NA

    summary["three_month_rolling_avg_recalls"] = (
        summary["total_recalls"]
        .astype(float)
        .rolling(
            window=ROLLING_WINDOW_MONTHS,
            min_periods=1,
        )
        .mean()
        .round(2)
    )

    summary = summary.reset_index(
        names="report_month_period"
    )

    summary["report_month"] = (
        summary["report_month_period"].astype(str)
    )

    output_columns = [
        "report_month",
        "total_recalls",
        "class_i_recalls",
        "class_ii_recalls",
        "class_iii_recalls",
        "not_yet_classified_recalls",
        "unique_firms",
        "median_reporting_lag_days",
        "termination_date_coverage_pct",
        "three_month_rolling_avg_recalls",
    ]

    return summary[output_columns]


def build_firm_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build recall metrics for each normalized recalling firm."""

    analytical = _prepare_analytics_base(frame)

    if analytical.empty:
        return pd.DataFrame(
            columns=[
                "recalling_firm_normalized",
                "recall_count",
                "unique_event_count",
                "class_i_count",
                "class_i_pct",
                "first_report_date",
                "latest_report_date",
                "median_reporting_lag_days",
                "active_year_count",
            ]
        )

    analytical["recalling_firm_normalized"] = (
        analytical["recalling_firm_normalized"]
        .astype("string")
        .fillna("UNKNOWN FIRM")
    )

    analytical["report_year"] = (
        analytical["report_date"].dt.year
    )

    grouped = analytical.groupby(
        "recalling_firm_normalized",
        dropna=False,
        observed=False,
    )

    summary = grouped.agg(
        recall_count=("record_id", "size"),
        unique_event_count=(
            "event_id",
            lambda values: int(
                values.dropna().nunique()
            ),
        ),
        class_i_count=(
            "classification",
            lambda values: int(
                values.eq("Class I").sum()
            ),
        ),
        first_report_date=("report_date", "min"),
        latest_report_date=("report_date", "max"),
        median_reporting_lag_days=(
            "reporting_lag_days",
            "median",
        ),
        active_year_count=(
            "report_year",
            lambda values: int(
                values.dropna().nunique()
            ),
        ),
    ).reset_index()

    summary["class_i_pct"] = (
        summary["class_i_count"]
        .div(summary["recall_count"])
        .mul(100)
        .round(2)
    )

    integer_columns = [
        "recall_count",
        "unique_event_count",
        "class_i_count",
        "active_year_count",
    ]

    for column in integer_columns:
        summary[column] = summary[column].astype("Int64")

    summary = summary.sort_values(
        by=[
            "recall_count",
            "recalling_firm_normalized",
        ],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    output_columns = [
        "recalling_firm_normalized",
        "recall_count",
        "unique_event_count",
        "class_i_count",
        "class_i_pct",
        "first_report_date",
        "latest_report_date",
        "median_reporting_lag_days",
        "active_year_count",
    ]

    return summary[output_columns]


def build_analytics_outputs(
    frame: pd.DataFrame,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build all analytical output tables.

    Returns:
        A tuple containing:
        1. Monthly summary.
        2. Firm summary.
    """

    monthly_summary = build_monthly_summary(
        frame,
        start_date=start_date,
        end_date=end_date,
    )

    firm_summary = build_firm_summary(frame)

    return monthly_summary, firm_summary