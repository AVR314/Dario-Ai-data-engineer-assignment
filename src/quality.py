"""Data-quality assessment for transformed drug recall records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.config import EXPECTED_CLASSIFICATIONS


REQUIRED_TRANSFORMED_COLUMNS = {
    "record_id",
    "source_record_hash",
    "recall_number",
    "classification",
    "severity_rank",
    "recalling_firm_raw",
    "product_description",
    "reason_for_recall",
    "recall_initiation_date",
    "report_date",
    "termination_date",
    "is_exact_duplicate",
    "invalid_recall_initiation_date_flag",
    "invalid_report_date_flag",
    "invalid_termination_date_flag",
    "negative_reporting_lag_flag",
    "negative_termination_duration_flag",
}

CRITICAL_BUSINESS_FIELDS = [
    "recall_number",
    "classification",
    "recalling_firm_raw",
    "product_description",
    "reason_for_recall",
    "recall_initiation_date",
    "report_date",
]


class DataQualityError(RuntimeError):
    """Raised when a quality report cannot be created safely."""


def _percentage(numerator: int, denominator: int) -> float:
    """Return a rounded percentage while handling empty datasets."""

    if denominator == 0:
        return 0.0

    return round((numerator / denominator) * 100, 2)


def _count_true(frame: pd.DataFrame, column: str) -> int:
    """Count truthy values in a nullable Boolean column."""

    if column not in frame.columns:
        return 0

    return int(frame[column].fillna(False).astype(bool).sum())


def _missing_field_summary(
    frame: pd.DataFrame,
    field: str,
) -> dict[str, Any]:
    """Summarize missingness for one transformed field."""

    total_records = len(frame)
    missing_count = int(frame[field].isna().sum())

    return {
        "field": field,
        "missing_count": missing_count,
        "available_count": total_records - missing_count,
        "missing_pct": _percentage(missing_count, total_records),
        "coverage_pct": _percentage(
            total_records - missing_count,
            total_records,
        ),
    }


def _append_issue(
    issues: list[dict[str, Any]],
    *,
    severity: str,
    code: str,
    message: str,
    affected_records: int = 0,
) -> None:
    """Append one structured quality finding."""

    issues.append(
        {
            "severity": severity,
            "code": code,
            "message": message,
            "affected_records": int(affected_records),
        }
    )


def build_data_quality_report(
    frame: pd.DataFrame,
    extraction_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a JSON-serializable quality report for transformed recall data.

    The report distinguishes between:
    - Technical failures
    - Genuine quality warnings
    - Expected or contextual missingness
    """

    if not isinstance(frame, pd.DataFrame):
        raise DataQualityError(
            "The quality layer requires a pandas DataFrame."
        )

    if extraction_metadata is not None and not isinstance(
        extraction_metadata,
        dict,
    ):
        raise DataQualityError(
            "Extraction metadata must be a dictionary when provided."
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    total_records = len(frame)

    missing_columns = sorted(
        REQUIRED_TRANSFORMED_COLUMNS.difference(frame.columns)
    )

    issues: list[dict[str, Any]] = []

    if missing_columns:
        _append_issue(
            issues,
            severity="error",
            code="missing_required_columns",
            message=(
                "The transformed dataset is missing required columns: "
                + ", ".join(missing_columns)
            ),
        )

        return {
            "generated_at_utc": generated_at,
            "overall_status": "failed",
            "total_records": total_records,
            "schema": {
                "required_column_count": len(
                    REQUIRED_TRANSFORMED_COLUMNS
                ),
                "missing_required_columns": missing_columns,
                "schema_valid": False,
            },
            "issues": issues,
            "summary": {
                "error_count": 1,
                "warning_count": 0,
                "info_count": 0,
            },
        }

    if total_records == 0:
        _append_issue(
            issues,
            severity="warning",
            code="empty_dataset",
            message=(
                "The extraction completed with zero records. "
                "No analytical metrics can be calculated."
            ),
        )

    critical_missingness = {
        field: _missing_field_summary(frame, field)
        for field in CRITICAL_BUSINESS_FIELDS
    }

    for field, summary in critical_missingness.items():
        if summary["missing_count"] > 0:
            _append_issue(
                issues,
                severity="warning",
                code=f"missing_{field}",
                message=(
                    f"{summary['missing_count']} records are missing "
                    f"the critical field '{field}'."
                ),
                affected_records=summary["missing_count"],
            )

    termination_summary = _missing_field_summary(
        frame,
        "termination_date",
    )

    _append_issue(
        issues,
        severity="info",
        code="termination_date_contextual_missingness",
        message=(
            "Termination-date coverage is reported separately because "
            "missing termination dates may be expected or structural. "
            "They are not automatically treated as data-quality failures."
        ),
        affected_records=termination_summary["missing_count"],
    )

    record_id_duplicate_count = int(
        frame["record_id"].duplicated(keep=False).sum()
    )

    exact_duplicate_rows = _count_true(
        frame,
        "is_exact_duplicate",
    )

    exact_duplicate_groups = int(
        frame.loc[
            frame["is_exact_duplicate"].fillna(False),
            "source_record_hash",
        ].nunique()
    )

    if record_id_duplicate_count > 0:
        _append_issue(
            issues,
            severity="error",
            code="duplicate_record_id",
            message=(
                "record_id must be unique after transformation, "
                "but duplicate values were found."
            ),
            affected_records=record_id_duplicate_count,
        )

    if exact_duplicate_rows > 0:
        _append_issue(
            issues,
            severity="warning",
            code="exact_source_duplicates",
            message=(
                f"{exact_duplicate_rows} rows belong to "
                f"{exact_duplicate_groups} exact duplicate groups. "
                "The rows were preserved and flagged for review."
            ),
            affected_records=exact_duplicate_rows,
        )

    non_missing_recall_numbers = frame[
        frame["recall_number"].notna()
    ]

    repeated_recall_mask = (
        non_missing_recall_numbers["recall_number"]
        .duplicated(keep=False)
    )

    repeated_recall_rows = int(repeated_recall_mask.sum())

    repeated_recall_number_count = int(
        non_missing_recall_numbers.loc[
            repeated_recall_mask,
            "recall_number",
        ].nunique()
    )

    if repeated_recall_rows > 0:
        _append_issue(
            issues,
            severity="info",
            code="repeated_recall_numbers",
            message=(
                f"{repeated_recall_number_count} recall numbers appear "
                f"across {repeated_recall_rows} rows. "
                "These records were not removed automatically."
            ),
            affected_records=repeated_recall_rows,
        )

    invalid_date_counts = {
        "recall_initiation_date": _count_true(
            frame,
            "invalid_recall_initiation_date_flag",
        ),
        "report_date": _count_true(
            frame,
            "invalid_report_date_flag",
        ),
        "termination_date": _count_true(
            frame,
            "invalid_termination_date_flag",
        ),
    }

    for field, invalid_count in invalid_date_counts.items():
        if invalid_count > 0:
            _append_issue(
                issues,
                severity="warning",
                code=f"invalid_{field}",
                message=(
                    f"{invalid_count} records contained a non-empty "
                    f"but invalid value in '{field}'."
                ),
                affected_records=invalid_count,
            )

    negative_reporting_lag_count = _count_true(
        frame,
        "negative_reporting_lag_flag",
    )

    negative_termination_duration_count = _count_true(
        frame,
        "negative_termination_duration_flag",
    )

    if negative_reporting_lag_count > 0:
        _append_issue(
            issues,
            severity="warning",
            code="negative_reporting_lag",
            message=(
                "Some report dates occur before recall initiation dates. "
                "Their reporting_lag_days values were excluded."
            ),
            affected_records=negative_reporting_lag_count,
        )

    if negative_termination_duration_count > 0:
        _append_issue(
            issues,
            severity="warning",
            code="negative_termination_duration",
            message=(
                "Some termination dates occur before recall initiation "
                "dates. Their termination_days values were excluded."
            ),
            affected_records=negative_termination_duration_count,
        )

    observed_classifications = set(
        frame["classification"].dropna().astype(str).unique()
    )

    unexpected_classifications = sorted(
        observed_classifications.difference(
            EXPECTED_CLASSIFICATIONS
        )
    )

    unexpected_classification_rows = int(
        frame["classification"]
        .isin(unexpected_classifications)
        .sum()
    )

    if unexpected_classifications:
        _append_issue(
            issues,
            severity="warning",
            code="unexpected_classifications",
            message=(
                "Unexpected recall classifications were preserved "
                "for review: "
                + ", ".join(unexpected_classifications)
            ),
            affected_records=unexpected_classification_rows,
        )

    unmapped_classification_count = int(
        (
            frame["classification"].notna()
            & frame["severity_rank"].isna()
        ).sum()
    )

    if unmapped_classification_count > 0:
        _append_issue(
            issues,
            severity="warning",
            code="classification_join_unmapped",
            message=(
                "Some classifications did not match the classification "
                "dimension table."
            ),
            affected_records=unmapped_classification_count,
        )

    extracted_count = None
    row_count_matches_extraction = None

    if extraction_metadata is not None:
        extracted_count = extraction_metadata.get(
            "records_extracted"
        )

        if extracted_count is not None:
            try:
                extracted_count = int(extracted_count)
            except (TypeError, ValueError):
                _append_issue(
                    issues,
                    severity="warning",
                    code="invalid_extraction_record_count",
                    message=(
                        "Extraction metadata contains an invalid "
                        "records_extracted value."
                    ),
                )
                extracted_count = None

        if extracted_count is not None:
            row_count_matches_extraction = (
                extracted_count == total_records
            )

            if not row_count_matches_extraction:
                _append_issue(
                    issues,
                    severity="warning",
                    code="extraction_transformation_count_mismatch",
                    message=(
                        "The transformed row count does not match "
                        "the extraction metadata."
                    ),
                    affected_records=abs(
                        total_records - extracted_count
                    ),
                )

    error_count = sum(
        issue["severity"] == "error"
        for issue in issues
    )

    warning_count = sum(
        issue["severity"] == "warning"
        for issue in issues
    )

    info_count = sum(
        issue["severity"] == "info"
        for issue in issues
    )

    if error_count > 0:
        overall_status = "failed"
    elif warning_count > 0:
        overall_status = "passed_with_warnings"
    else:
        overall_status = "passed"

    return {
        "generated_at_utc": generated_at,
        "overall_status": overall_status,
        "total_records": total_records,
        "schema": {
            "required_column_count": len(
                REQUIRED_TRANSFORMED_COLUMNS
            ),
            "missing_required_columns": [],
            "schema_valid": True,
        },
        "completeness": {
            "critical_fields": critical_missingness,
            "termination_date": {
                **termination_summary,
                "interpretation": (
                    "Contextual missingness. Duration metrics must use "
                    "only records with a valid termination date."
                ),
            },
        },
        "duplicates": {
            "record_id_duplicate_rows": record_id_duplicate_count,
            "exact_duplicate_rows": exact_duplicate_rows,
            "exact_duplicate_groups": exact_duplicate_groups,
            "repeated_recall_number_rows": repeated_recall_rows,
            "repeated_recall_number_count": (
                repeated_recall_number_count
            ),
        },
        "dates": {
            "invalid_date_counts": invalid_date_counts,
            "negative_reporting_lag_count": (
                negative_reporting_lag_count
            ),
            "negative_termination_duration_count": (
                negative_termination_duration_count
            ),
        },
        "classifications": {
            "observed_values": sorted(
                observed_classifications
            ),
            "unexpected_values": unexpected_classifications,
            "unexpected_record_count": (
                unexpected_classification_rows
            ),
            "unmapped_record_count": (
                unmapped_classification_count
            ),
        },
        "reconciliation": {
            "records_extracted": extracted_count,
            "records_transformed": total_records,
            "row_count_matches_extraction": (
                row_count_matches_extraction
            ),
        },
        "issues": issues,
        "summary": {
            "error_count": int(error_count),
            "warning_count": int(warning_count),
            "info_count": int(info_count),
        },
    }