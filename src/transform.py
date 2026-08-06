"""Transformation logic for openFDA drug recall records."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd


SOURCE_FIELDS = [
    "recall_number",
    "event_id",
    "classification",
    "recalling_firm",
    "product_description",
    "reason_for_recall",
    "voluntary_mandated",
    "distribution_pattern",
    "city",
    "state",
    "country",
    "recall_initiation_date",
    "report_date",
    "termination_date",
    "status",
]

DATE_FIELDS = [
    "recall_initiation_date",
    "report_date",
    "termination_date",
]

TEXT_FIELDS = [
    "recall_number",
    "event_id",
    "classification",
    "recalling_firm",
    "product_description",
    "reason_for_recall",
    "voluntary_mandated",
    "distribution_pattern",
    "city",
    "state",
    "country",
    "status",
]


class TransformationError(RuntimeError):
    """Raised when raw recall records cannot be transformed safely."""


def build_classification_dimension() -> pd.DataFrame:
    """Create the recall-classification lookup table."""

    rows = [
        {
            "classification": "Class I",
            "severity_rank": 3,
            "severity_label": "High",
            "classification_description": (
                "The highest recall severity, involving a reasonable "
                "probability of serious health consequences."
            ),
            "display_order": 1,
        },
        {
            "classification": "Class II",
            "severity_rank": 2,
            "severity_label": "Moderate",
            "classification_description": (
                "A recall involving possible temporary or medically "
                "reversible health consequences."
            ),
            "display_order": 2,
        },
        {
            "classification": "Class III",
            "severity_rank": 1,
            "severity_label": "Low",
            "classification_description": (
                "A recall where adverse health consequences are considered "
                "unlikely."
            ),
            "display_order": 3,
        },
        {
            "classification": "Not Yet Classified",
            "severity_rank": 0,
            "severity_label": "Unclassified",
            "classification_description": (
                "The recall had not received a final classification when "
                "the record was published."
            ),
            "display_order": 4,
        },
    ]

    return pd.DataFrame(rows)


def _clean_text_value(value: Any) -> Any:
    """Trim whitespace and convert empty textual values into missing values."""

    if value is None or pd.isna(value):
        return pd.NA

    cleaned = re.sub(r"\s+", " ", str(value)).strip()

    return cleaned if cleaned else pd.NA


def normalize_firm_name(value: Any) -> Any:
    """
    Normalize a company name conservatively.

    The function standardizes casing, punctuation and whitespace. It does not
    use fuzzy matching or remove legal suffixes, because doing so could merge
    different companies incorrectly.
    """

    cleaned = _clean_text_value(value)

    if pd.isna(cleaned):
        return pd.NA

    normalized = str(cleaned).upper()
    normalized = re.sub(r"[^A-Z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized if normalized else pd.NA


def _parse_date_series(
    raw_series: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Parse YYYYMMDD values and return parsed dates plus invalid-value flags.

    Missing source values are not considered invalid. A non-empty value that
    cannot be parsed is marked as invalid.
    """

    cleaned = raw_series.map(_clean_text_value).astype("string")

    parsed = pd.to_datetime(
        cleaned,
        format="%Y%m%d",
        errors="coerce",
    )

    invalid = cleaned.notna() & parsed.isna()

    return parsed, invalid


def _create_source_hash(frame: pd.DataFrame) -> pd.Series:
    """Create a deterministic hash from all relevant source fields."""

    hash_input = (
        frame[SOURCE_FIELDS]
        .fillna("")
        .astype(str)
        .agg("\x1f".join, axis=1)
    )

    return hash_input.map(
        lambda value: hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()[:20]
    )


def _add_record_identifiers(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add deterministic record identifiers without deleting duplicate records.

    Identical records share the same source hash. An occurrence suffix keeps
    record_id unique while preserving every source row for quality review.
    """

    transformed = frame.copy()

    transformed["source_record_hash"] = _create_source_hash(transformed)

    occurrence_number = (
        transformed.groupby(
            "source_record_hash",
            sort=False,
        )
        .cumcount()
        .add(1)
    )

    transformed["record_id"] = (
        transformed["source_record_hash"]
        + "-"
        + occurrence_number.astype(str).str.zfill(3)
    )

    transformed["is_exact_duplicate"] = (
        transformed["source_record_hash"].duplicated(keep=False)
    )

    recall_counts = (
        transformed.groupby(
            "recall_number",
            dropna=False,
        )["record_id"]
        .transform("size")
        .astype("Int64")
    )

    transformed["recall_number_occurrence_count"] = recall_counts
    transformed["has_repeated_recall_number"] = recall_counts.gt(1)

    return transformed


def _add_derived_date_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Add safe date-derived metrics and anomaly flags."""

    transformed = frame.copy()

    reporting_lag = (
        transformed["report_date"]
        - transformed["recall_initiation_date"]
    ).dt.days

    transformed["negative_reporting_lag_flag"] = reporting_lag.lt(0)

    transformed["reporting_lag_days"] = (
        reporting_lag
        .mask(reporting_lag.lt(0))
        .astype("Int64")
    )

    termination_duration = (
        transformed["termination_date"]
        - transformed["recall_initiation_date"]
    ).dt.days

    transformed["negative_termination_duration_flag"] = (
        termination_duration.lt(0)
    )

    transformed["termination_days"] = (
        termination_duration
        .mask(termination_duration.lt(0))
        .astype("Int64")
    )

    transformed["has_termination_date"] = (
        transformed["termination_date"].notna()
    )

    transformed["report_month"] = (
        transformed["report_date"]
        .dt.to_period("M")
        .astype("string")
        .replace("NaT", pd.NA)
    )

    return transformed


def transform_recalls(
    records: list[dict[str, Any]],
    source_extract_timestamp: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert raw openFDA records into an enriched analytical table.

    Returns:
        A tuple containing:
        1. The enriched recall-level table.
        2. The classification dimension table.
    """

    if not isinstance(records, list):
        raise TransformationError("Raw records must be provided as a list.")

    if not all(isinstance(record, dict) for record in records):
        raise TransformationError(
            "Every raw record must be represented as a dictionary."
        )

    dimension = build_classification_dimension()

    if not records:
        empty_columns = [
            "record_id",
            "source_record_hash",
            *SOURCE_FIELDS,
            "recalling_firm_raw",
            "recalling_firm_normalized",
            "status_at_publication",
            "severity_rank",
            "severity_label",
            "classification_description",
            "reporting_lag_days",
            "termination_days",
            "has_termination_date",
            "report_month",
            "source_extract_timestamp",
        ]

        return pd.DataFrame(columns=empty_columns), dimension

    frame = pd.DataFrame(records)

    for field in SOURCE_FIELDS:
        if field not in frame.columns:
            frame[field] = pd.NA

    frame = frame[SOURCE_FIELDS].copy()

    for field in TEXT_FIELDS:
        frame[field] = frame[field].map(_clean_text_value).astype("string")

    frame["recalling_firm_raw"] = frame["recalling_firm"]
    frame["recalling_firm_normalized"] = (
        frame["recalling_firm_raw"].map(normalize_firm_name).astype("string")
    )

    frame["status_at_publication"] = frame["status"]

    for field in DATE_FIELDS:
        raw_values = frame[field].copy()

        parsed_dates, invalid_flags = _parse_date_series(raw_values)

        frame[field] = parsed_dates
        frame[f"invalid_{field}_flag"] = invalid_flags

    frame = _add_record_identifiers(frame)
    frame = _add_derived_date_fields(frame)

    frame = frame.merge(
        dimension,
        on="classification",
        how="left",
        validate="many_to_one",
    )

    if source_extract_timestamp is None:
        source_extract_timestamp = datetime.now(timezone.utc).isoformat()

    parsed_extract_timestamp = pd.to_datetime(
        source_extract_timestamp,
        utc=True,
        errors="coerce",
    )

    if pd.isna(parsed_extract_timestamp):
        raise TransformationError(
            "source_extract_timestamp must be a valid datetime value."
        )

    frame["source_extract_timestamp"] = parsed_extract_timestamp

    output_columns = [
        "record_id",
        "source_record_hash",
        "recall_number",
        "event_id",
        "classification",
        "severity_rank",
        "severity_label",
        "classification_description",
        "recalling_firm_raw",
        "recalling_firm_normalized",
        "product_description",
        "reason_for_recall",
        "voluntary_mandated",
        "distribution_pattern",
        "city",
        "state",
        "country",
        "recall_initiation_date",
        "report_date",
        "termination_date",
        "status_at_publication",
        "reporting_lag_days",
        "termination_days",
        "has_termination_date",
        "report_month",
        "is_exact_duplicate",
        "recall_number_occurrence_count",
        "has_repeated_recall_number",
        "invalid_recall_initiation_date_flag",
        "invalid_report_date_flag",
        "invalid_termination_date_flag",
        "negative_reporting_lag_flag",
        "negative_termination_duration_flag",
        "source_extract_timestamp",
    ]

    return frame[output_columns], dimension