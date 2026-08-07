"""Tests for the end-to-end ETL orchestration layer."""

import unittest
from unittest.mock import patch

import pandas as pd

import etl
from etl import (
    ETLPipelineError,
    extract_records,
    resolve_extract_timestamp,
    resolve_quality_status,
    run_etl,
)
from src.api_client import OpenFDAClientError


class ETLPipelineTests(unittest.TestCase):
    """Validate live extraction, cache fallback and ETL control flow."""

    def test_resolve_extract_timestamp_uses_first_available_value(
        self,
    ) -> None:
        """The most relevant extraction timestamp should be selected."""

        metadata = {
            "snapshot_saved_at_utc": "2026-08-06T10:00:00+00:00",
            "extracted_at_utc": "2026-08-06T09:00:00+00:00",
        }

        result = resolve_extract_timestamp(metadata)

        self.assertEqual(
            result,
            "2026-08-06T09:00:00+00:00",
        )

    def test_resolve_quality_status_from_summary(
        self,
    ) -> None:
        """A nested quality status should be read correctly."""

        report = {
            "summary": {
                "status": "passed_with_warnings",
            }
        }

        result = resolve_quality_status(report)

        self.assertEqual(
            result,
            "passed_with_warnings",
        )

    @patch("etl.save_raw_snapshot")
    @patch("etl.fetch_drug_recalls")
    def test_live_extraction_saves_raw_snapshot(
        self,
        mock_fetch_drug_recalls,
        mock_save_raw_snapshot,
    ) -> None:
        """A successful live extraction should be stored locally."""

        records = [
            {
                "recall_number": "D-1001-2024",
            }
        ]

        mock_fetch_drug_recalls.return_value = (
            records,
            {
                "records_extracted": 1,
                "extracted_at_utc": (
                    "2026-08-06T12:00:00+00:00"
                ),
            },
        )

        returned_records, metadata = extract_records()

        self.assertEqual(
            returned_records,
            records,
        )
        self.assertEqual(
            metadata["extraction_status"],
            "live",
        )
        self.assertFalse(
            metadata["used_cached_data"]
        )

        mock_save_raw_snapshot.assert_called_once_with(
            records,
            metadata,
        )

    @patch("etl.load_cached_raw_snapshot")
    @patch("etl.raw_snapshot_exists")
    @patch("etl.fetch_drug_recalls")
    def test_api_failure_uses_cached_snapshot(
        self,
        mock_fetch_drug_recalls,
        mock_raw_snapshot_exists,
        mock_load_cached_raw_snapshot,
    ) -> None:
        """A live API failure should use an available local snapshot."""

        cached_records = [
            {
                "recall_number": "D-CACHED-2024",
            }
        ]

        cached_metadata = {
            "records_extracted": 1,
            "used_cached_data": True,
            "extraction_status": "cached_fallback",
        }

        mock_fetch_drug_recalls.side_effect = OpenFDAClientError(
            "API unavailable"
        )

        mock_raw_snapshot_exists.return_value = True

        mock_load_cached_raw_snapshot.return_value = (
            cached_records,
            cached_metadata,
        )

        records, metadata = extract_records()

        self.assertEqual(
            records,
            cached_records,
        )
        self.assertEqual(
            metadata,
            cached_metadata,
        )

        mock_load_cached_raw_snapshot.assert_called_once()

        fallback_reason = (
            mock_load_cached_raw_snapshot
            .call_args
            .kwargs["fallback_reason"]
        )

        self.assertIn(
            "OpenFDAClientError",
            fallback_reason,
        )
        self.assertIn(
            "API unavailable",
            fallback_reason,
        )

    @patch("etl.raw_snapshot_exists")
    @patch("etl.fetch_drug_recalls")
    def test_api_failure_without_cache_raises_error(
        self,
        mock_fetch_drug_recalls,
        mock_raw_snapshot_exists,
    ) -> None:
        """The ETL must fail clearly if neither API nor cache is available."""

        mock_fetch_drug_recalls.side_effect = OpenFDAClientError(
            "API unavailable"
        )

        mock_raw_snapshot_exists.return_value = False

        with self.assertRaisesRegex(
            ETLPipelineError,
            "no valid local raw snapshot",
        ):
            extract_records()

    @patch("etl.load_cached_raw_snapshot")
    @patch("etl.raw_snapshot_exists")
    @patch("etl.fetch_drug_recalls")
    def test_unexpected_extraction_error_does_not_use_cache(
        self,
        mock_fetch_drug_recalls,
        mock_raw_snapshot_exists,
        mock_load_cached_raw_snapshot,
    ) -> None:
        """Programming defects must propagate instead of using stale data."""

        mock_fetch_drug_recalls.side_effect = RuntimeError(
            "unexpected programming failure"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "unexpected programming failure",
        ):
            extract_records()

        mock_raw_snapshot_exists.assert_not_called()
        mock_load_cached_raw_snapshot.assert_not_called()

    @patch("etl.save_json_document")
    @patch("etl.save_dataframe_csv")
    @patch("etl.build_analytics_outputs")
    @patch("etl.build_data_quality_report")
    @patch("etl.transform_recalls")
    @patch("etl.extract_records")
    @patch("etl.ensure_data_directories")
    def test_successful_etl_builds_and_saves_all_outputs(
        self,
        mock_ensure_data_directories,
        mock_extract_records,
        mock_transform_recalls,
        mock_build_data_quality_report,
        mock_build_analytics_outputs,
        mock_save_dataframe_csv,
        mock_save_json_document,
    ) -> None:
        """A successful pipeline should create all analytical outputs."""

        records = [
            {
                "recall_number": "D-1001-2024",
            },
            {
                "recall_number": "D-1002-2024",
            },
        ]

        extraction_metadata = {
            "records_extracted": 2,
            "extracted_at_utc": (
                "2026-08-06T12:00:00+00:00"
            ),
            "extraction_status": "live",
            "used_cached_data": False,
        }

        recalls_frame = pd.DataFrame(
            {
                "record_id": [
                    "record-001",
                    "record-002",
                ]
            }
        )

        classification_dimension = pd.DataFrame(
            {
                "classification": [
                    "Class I",
                ]
            }
        )

        quality_report = {
            "overall_status": "passed",
            "summary": {
                "error_count": 0,
                "warning_count": 0,
                "info_count": 0,
            },
        }

        monthly_summary = pd.DataFrame(
            {
                "report_month": [
                    "2026-08",
                ]
            }
        )

        firm_summary = pd.DataFrame(
            {
                "recalling_firm_normalized": [
                    "EXAMPLE PHARMA",
                ]
            }
        )

        mock_extract_records.return_value = (
            records,
            extraction_metadata,
        )

        mock_transform_recalls.return_value = (
            recalls_frame,
            classification_dimension,
        )

        mock_build_data_quality_report.return_value = (
            quality_report
        )

        execution_order: list[str] = []

        def build_analytics(*args, **kwargs):
            execution_order.append("analytics")
            return monthly_summary, firm_summary

        def save_csv(*args, **kwargs):
            self.assertNotIn(
                "pipeline_completed_at_utc",
                quality_report.get("pipeline_context", {}),
            )
            execution_order.append("csv")

        def save_quality(*args, **kwargs):
            self.assertIn(
                "pipeline_completed_at_utc",
                quality_report["pipeline_context"],
            )
            execution_order.append("quality")

        mock_build_analytics_outputs.side_effect = build_analytics
        mock_save_dataframe_csv.side_effect = save_csv
        mock_save_json_document.side_effect = save_quality

        result = run_etl()

        self.assertEqual(
            result["status"],
            "success",
        )
        self.assertEqual(
            result["quality_status"],
            "passed",
        )
        self.assertEqual(
            result["raw_record_count"],
            2,
        )
        self.assertEqual(
            result["transformed_record_count"],
            2,
        )
        self.assertEqual(
            result["monthly_summary_row_count"],
            1,
        )
        self.assertEqual(
            result["firm_summary_row_count"],
            1,
        )
        self.assertFalse(
            result["used_cached_data"]
        )

        mock_ensure_data_directories.assert_called_once()

        mock_build_analytics_outputs.assert_called_once()

        self.assertEqual(
            mock_save_dataframe_csv.call_count,
            4,
        )

        mock_save_json_document.assert_called_once()

        self.assertEqual(
            execution_order,
            [
                "analytics",
                "csv",
                "csv",
                "csv",
                "csv",
                "quality",
            ],
        )

        self.assertIn(
            "pipeline_context",
            quality_report,
        )

    @patch("etl.save_json_document")
    @patch("etl.save_dataframe_csv")
    @patch("etl.build_analytics_outputs")
    @patch("etl.build_data_quality_report")
    @patch("etl.transform_recalls")
    @patch("etl.extract_records")
    @patch("etl.ensure_data_directories")
    def test_failed_quality_stops_before_analytics(
        self,
        mock_ensure_data_directories,
        mock_extract_records,
        mock_transform_recalls,
        mock_build_data_quality_report,
        mock_build_analytics_outputs,
        mock_save_dataframe_csv,
        mock_save_json_document,
    ) -> None:
        """Critical quality failures must stop analytical output creation."""

        records = [
            {
                "recall_number": "D-1001-2024",
            }
        ]

        metadata = {
            "records_extracted": 1,
            "extracted_at_utc": (
                "2026-08-06T12:00:00+00:00"
            ),
            "extraction_status": "live",
            "used_cached_data": False,
        }

        recalls_frame = pd.DataFrame(
            {
                "record_id": [
                    "record-001",
                ]
            }
        )

        classification_dimension = pd.DataFrame(
            {
                "classification": [
                    "Class I",
                ]
            }
        )

        mock_extract_records.return_value = (
            records,
            metadata,
        )

        mock_transform_recalls.return_value = (
            recalls_frame,
            classification_dimension,
        )

        mock_build_data_quality_report.return_value = {
            "overall_status": "failed",
            "issues": [
                {
                    "severity": "error",
                    "code": "missing_required_columns",
                    "message": "Required columns are missing.",
                    "affected_records": 0,
                }
            ],
            "summary": {
                "error_count": 1,
                "warning_count": 0,
                "info_count": 0,
            },
        }

        with self.assertRaisesRegex(
            ETLPipelineError,
            "failed critical data quality",
        ):
            run_etl()

        mock_build_analytics_outputs.assert_not_called()
        mock_save_dataframe_csv.assert_not_called()
        mock_save_json_document.assert_not_called()

    @patch("etl.save_json_document")
    @patch("etl.save_dataframe_csv")
    @patch("etl.build_analytics_outputs")
    @patch("etl.build_data_quality_report")
    @patch("etl.transform_recalls")
    @patch("etl.extract_records")
    @patch("etl.ensure_data_directories")
    def test_analytics_failure_does_not_save_processed_outputs(
        self,
        mock_ensure_data_directories,
        mock_extract_records,
        mock_transform_recalls,
        mock_build_data_quality_report,
        mock_build_analytics_outputs,
        mock_save_dataframe_csv,
        mock_save_json_document,
    ) -> None:
        """Failed analytics must not partially replace processed outputs."""

        records = [{"recall_number": "D-1001-2024"}]
        metadata = {
            "records_extracted": 1,
            "extracted_at_utc": "2026-08-06T12:00:00+00:00",
            "extraction_status": "live",
            "used_cached_data": False,
        }
        recalls_frame = pd.DataFrame({"record_id": ["record-001"]})
        classification_dimension = pd.DataFrame(
            {"classification": ["Class I"]}
        )

        mock_extract_records.return_value = records, metadata
        mock_transform_recalls.return_value = (
            recalls_frame,
            classification_dimension,
        )
        mock_build_data_quality_report.return_value = {
            "overall_status": "passed",
            "issues": [],
            "summary": {
                "error_count": 0,
                "warning_count": 0,
                "info_count": 0,
            },
        }
        mock_build_analytics_outputs.side_effect = RuntimeError(
            "analytics failed"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "analytics failed",
        ):
            run_etl()

        mock_save_dataframe_csv.assert_not_called()
        mock_save_json_document.assert_not_called()


if __name__ == "__main__":
    unittest.main()
