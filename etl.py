"""Main ETL entry point for the Drug Recall Intelligence Dashboard."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.analytics import build_analytics_outputs
from src.api_client import OpenFDAClientError, fetch_drug_recalls
from src.config import (
    CLASSIFICATION_DIM_FILE,
    DATA_QUALITY_REPORT_FILE,
    END_DATE,
    FIRM_SUMMARY_FILE,
    MONTHLY_SUMMARY_FILE,
    RECALLS_OUTPUT_FILE,
    START_DATE,
)
from src.quality import build_data_quality_report
from src.storage import (
    ensure_data_directories,
    load_cached_raw_snapshot,
    raw_snapshot_exists,
    save_dataframe_csv,
    save_json_document,
    save_raw_snapshot,
)
from src.transform import transform_recalls


LOGGER = logging.getLogger(__name__)


class ETLPipelineError(RuntimeError):
    """Raised when the end-to-end ETL pipeline cannot complete safely."""


def configure_logging() -> None:
    """Configure readable console logging for local ETL execution."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_extract_timestamp(
    metadata: dict[str, Any],
) -> str:
    """
    Resolve the source extraction timestamp from available metadata.

    The API and cached-data paths may expose slightly different timestamp
    fields, so the function checks known candidates in a stable order.
    """

    timestamp_candidates = [
        "extracted_at_utc",
        "extraction_completed_at_utc",
        "extraction_timestamp_utc",
        "snapshot_saved_at_utc",
    ]

    for field_name in timestamp_candidates:
        value = metadata.get(field_name)

        if value:
            return str(value)

    return datetime.now(timezone.utc).isoformat()


def resolve_quality_status(
    quality_report: dict[str, Any],
) -> str:
    """Read the quality status without depending on one report layout."""

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
        if value is not None:
            return str(value).strip().lower()

    return "unknown"


def extract_records() -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    """
    Extract current records from openFDA.

    If the live API call fails and a valid local snapshot exists, the pipeline
    continues with the cached snapshot and records that fallback in metadata.
    """

    LOGGER.info(
        "Extracting openFDA drug recall records from %s to %s.",
        START_DATE,
        END_DATE,
    )

    try:
        records, metadata = fetch_drug_recalls(
            start_date=START_DATE,
            end_date=END_DATE,
        )

    except OpenFDAClientError as exc:
        fallback_reason = (
            f"{type(exc).__name__}: {exc}"
        )

        LOGGER.warning(
            "Live extraction failed: %s",
            fallback_reason,
        )

        if not raw_snapshot_exists():
            raise ETLPipelineError(
                "Live extraction failed and no valid local raw "
                "snapshot is available."
            ) from exc

        LOGGER.info(
            "Loading the latest local raw snapshot."
        )

        records, metadata = load_cached_raw_snapshot(
            fallback_reason=fallback_reason,
        )

        return records, metadata

    metadata = {
        **metadata,
        "used_cached_data": False,
        "extraction_status": "live",
    }

    save_raw_snapshot(
        records,
        metadata,
    )

    LOGGER.info(
        "Saved %s raw records locally.",
        len(records),
    )

    return records, metadata


def run_etl() -> dict[str, Any]:
    """Run extraction, transformation, validation and analytics."""

    ensure_data_directories()

    pipeline_started_at = (
        datetime.now(timezone.utc)
    )

    LOGGER.info(
        "Starting the drug recall ETL pipeline."
    )

    records, extraction_metadata = extract_records()

    extract_timestamp = resolve_extract_timestamp(
        extraction_metadata
    )

    LOGGER.info(
        "Transforming %s raw records.",
        len(records),
    )

    recalls_frame, classification_dimension = (
        transform_recalls(
            records,
            source_extract_timestamp=extract_timestamp,
        )
    )

    LOGGER.info(
        "Running data quality checks."
    )

    quality_report = build_data_quality_report(
        recalls_frame,
        extraction_metadata=extraction_metadata,
    )

    quality_report["pipeline_context"] = {
        "pipeline_started_at_utc": (
            pipeline_started_at.isoformat()
        ),
        "requested_start_date": START_DATE,
        "requested_end_date": END_DATE,
        "extraction_status": extraction_metadata.get(
            "extraction_status",
            "unknown",
        ),
        "used_cached_data": bool(
            extraction_metadata.get(
                "used_cached_data",
                False,
            )
        ),
        "raw_record_count": len(records),
        "transformed_record_count": len(
            recalls_frame
        ),
    }

    quality_status = resolve_quality_status(
        quality_report
    )

    LOGGER.info(
        "Data quality status: %s",
        quality_status,
    )

    if quality_status == "failed":
        issues = quality_report.get("issues", [])
        issue_details = "; ".join(
            f"{issue.get('code', 'unknown')}: "
            f"{issue.get('message', 'No details provided.')}"
            for issue in issues[:5]
            if isinstance(issue, dict)
        )

        if not issue_details:
            issue_details = "No structured issue details were provided."

        LOGGER.error(
            "Data quality validation failed. %s",
            issue_details,
        )

        raise ETLPipelineError(
            "The transformed dataset failed critical data quality "
            f"checks. Findings: {issue_details}"
        )

    LOGGER.info(
        "Building monthly and firm-level analytical summaries."
    )

    monthly_summary, firm_summary = (
        build_analytics_outputs(
            recalls_frame,
            start_date=START_DATE,
            end_date=END_DATE,
        )
    )

    # All calculations must succeed before any processed output is replaced.
    save_dataframe_csv(
        recalls_frame,
        RECALLS_OUTPUT_FILE,
    )

    save_dataframe_csv(
        classification_dimension,
        CLASSIFICATION_DIM_FILE,
    )

    save_dataframe_csv(
        monthly_summary,
        MONTHLY_SUMMARY_FILE,
    )

    save_dataframe_csv(
        firm_summary,
        FIRM_SUMMARY_FILE,
    )

    pipeline_completed_at = (
        datetime.now(timezone.utc)
    )

    quality_report["pipeline_context"][
        "pipeline_completed_at_utc"
    ] = pipeline_completed_at.isoformat()

    save_json_document(
        quality_report,
        DATA_QUALITY_REPORT_FILE,
    )

    duration_seconds = (
        pipeline_completed_at
        - pipeline_started_at
    ).total_seconds()

    result = {
        "status": "success",
        "quality_status": quality_status,
        "used_cached_data": bool(
            extraction_metadata.get(
                "used_cached_data",
                False,
            )
        ),
        "raw_record_count": len(records),
        "transformed_record_count": len(
            recalls_frame
        ),
        "monthly_summary_row_count": len(
            monthly_summary
        ),
        "firm_summary_row_count": len(
            firm_summary
        ),
        "duration_seconds": round(
            duration_seconds,
            2,
        ),
    }

    LOGGER.info(
        "ETL completed successfully in %.2f seconds.",
        duration_seconds,
    )

    LOGGER.info(
        "Created processed outputs in data/processed."
    )

    return result


def main() -> None:
    """Execute the ETL pipeline from the command line."""

    configure_logging()

    try:
        result = run_etl()

    except Exception:
        LOGGER.exception(
            "ETL pipeline failed."
        )
        raise

    print()
    print("ETL SUMMARY")
    print("-----------")
    print(
        f"Status: {result['status']}"
    )
    print(
        f"Quality status: {result['quality_status']}"
    )
    print(
        f"Used cached data: {result['used_cached_data']}"
    )
    print(
        f"Raw records: {result['raw_record_count']}"
    )
    print(
        "Transformed records: "
        f"{result['transformed_record_count']}"
    )
    print(
        "Monthly summary rows: "
        f"{result['monthly_summary_row_count']}"
    )
    print(
        "Firm summary rows: "
        f"{result['firm_summary_row_count']}"
    )
    print(
        f"Duration: {result['duration_seconds']} seconds"
    )


if __name__ == "__main__":
    main()
