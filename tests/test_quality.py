"""Tests for transformed recall data-quality reporting."""

import unittest
from typing import Any

import pandas as pd

from src.quality import (
    DataQualityError,
    build_data_quality_report,
)
from src.transform import transform_recalls


def make_recall(**overrides: Any) -> dict[str, Any]:
    """Create one valid recall record with optional overrides."""

    record: dict[str, Any] = {
        "recall_number": "D-1001-2024",
        "event_id": "90001",
        "classification": "Class II",
        "recalling_firm": "Example Pharma, Inc.",
        "product_description": "Example tablets",
        "reason_for_recall": "Labeling issue",
        "voluntary_mandated": "Voluntary: Firm initiated",
        "distribution_pattern": "Nationwide",
        "city": "Boston",
        "state": "MA",
        "country": "United States",
        "recall_initiation_date": "20240101",
        "report_date": "20240110",
        "termination_date": "20240201",
        "status": "Terminated",
    }

    record.update(overrides)
    return record


def transform(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Transform sample records using a stable extraction timestamp."""

    frame, _ = transform_recalls(
        records,
        source_extract_timestamp="2026-08-06T12:00:00+00:00",
    )

    return frame


def issue_codes(report: dict[str, Any]) -> set[str]:
    """Return all issue codes contained in a quality report."""

    return {
        str(issue["code"])
        for issue in report["issues"]
    }


class QualityTests(unittest.TestCase):
    """Validate schema, missingness, duplicate and date reporting."""

    def test_valid_dataset_passes_quality_checks(self) -> None:
        """A complete valid dataset should pass without warnings."""

        frame = transform([make_recall()])

        report = build_data_quality_report(
            frame,
            extraction_metadata={
                "records_reported_by_api": 1,
                "records_extracted": 1,
                "count_mismatch": False,
            },
        )

        self.assertEqual(report["overall_status"], "passed")
        self.assertTrue(report["schema"]["schema_valid"])
        self.assertEqual(report["total_records"], 1)
        self.assertEqual(report["summary"]["error_count"], 0)
        self.assertEqual(report["summary"]["warning_count"], 0)
        self.assertEqual(report["summary"]["info_count"], 1)
        self.assertTrue(
            report["reconciliation"][
                "row_count_matches_extraction"
            ]
        )
        self.assertTrue(
            report["reconciliation"][
                "api_reported_count_matches_extraction"
            ]
        )
        self.assertNotIn(
            "api_reported_count_mismatch",
            issue_codes(report),
        )

    def test_missing_required_columns_fail_report(self) -> None:
        """A structurally incomplete DataFrame should fail clearly."""

        incomplete_frame = pd.DataFrame(
            {
                "record_id": ["record-001"],
            }
        )

        report = build_data_quality_report(incomplete_frame)

        self.assertEqual(report["overall_status"], "failed")
        self.assertFalse(report["schema"]["schema_valid"])
        self.assertTrue(
            report["schema"]["missing_required_columns"]
        )
        self.assertIn(
            "missing_required_columns",
            issue_codes(report),
        )

    def test_missing_critical_business_field_is_warning(self) -> None:
        """Missing important business content should create a warning."""

        frame = transform(
            [
                make_recall(
                    product_description=None,
                )
            ]
        )

        report = build_data_quality_report(frame)

        product_summary = report["completeness"][
            "critical_fields"
        ]["product_description"]

        self.assertEqual(
            report["overall_status"],
            "passed_with_warnings",
        )
        self.assertEqual(product_summary["missing_count"], 1)
        self.assertEqual(product_summary["coverage_pct"], 0.0)
        self.assertIn(
            "missing_product_description",
            issue_codes(report),
        )

    def test_missing_termination_date_is_contextual_info(self) -> None:
        """A missing termination date must not fail quality by itself."""

        frame = transform(
            [
                make_recall(
                    termination_date=None,
                    status="Ongoing",
                )
            ]
        )

        report = build_data_quality_report(frame)

        termination_summary = report["completeness"][
            "termination_date"
        ]

        self.assertEqual(report["overall_status"], "passed")
        self.assertEqual(
            termination_summary["missing_count"],
            1,
        )
        self.assertEqual(
            termination_summary["coverage_pct"],
            0.0,
        )
        self.assertIn(
            "Contextual missingness",
            termination_summary["interpretation"],
        )
        self.assertIn(
            "termination_date_contextual_missingness",
            issue_codes(report),
        )

    def test_exact_duplicates_are_reported_but_preserved(self) -> None:
        """Exact source duplicates should create a quality warning."""

        duplicated_record = make_recall()

        frame = transform(
            [
                duplicated_record,
                duplicated_record.copy(),
            ]
        )

        report = build_data_quality_report(
            frame,
            extraction_metadata={"records_extracted": 2},
        )

        duplicate_summary = report["duplicates"]

        self.assertEqual(
            report["overall_status"],
            "passed_with_warnings",
        )
        self.assertEqual(
            duplicate_summary["exact_duplicate_rows"],
            2,
        )
        self.assertEqual(
            duplicate_summary["exact_duplicate_groups"],
            1,
        )
        self.assertEqual(
            duplicate_summary[
                "repeated_recall_number_rows"
            ],
            2,
        )
        self.assertIn(
            "exact_source_duplicates",
            issue_codes(report),
        )
        self.assertIn(
            "repeated_recall_numbers",
            issue_codes(report),
        )

    def test_unexpected_classification_is_preserved_and_warned(self) -> None:
        """Unknown classifications should remain visible for review."""

        frame = transform(
            [
                make_recall(
                    classification="Unexpected Class",
                )
            ]
        )

        report = build_data_quality_report(frame)

        classification_summary = report["classifications"]

        self.assertEqual(
            report["overall_status"],
            "passed_with_warnings",
        )
        self.assertEqual(
            classification_summary["unexpected_values"],
            ["Unexpected Class"],
        )
        self.assertEqual(
            classification_summary["unexpected_record_count"],
            1,
        )
        self.assertEqual(
            classification_summary["unmapped_record_count"],
            1,
        )
        self.assertIn(
            "unexpected_classifications",
            issue_codes(report),
        )
        self.assertIn(
            "classification_join_unmapped",
            issue_codes(report),
        )

    def test_invalid_and_negative_dates_are_reported(self) -> None:
        """Malformed and logically inconsistent dates should be warned."""

        frame = transform(
            [
                make_recall(
                    recall_initiation_date="20240210",
                    report_date="20240201",
                    termination_date="invalid-date",
                )
            ]
        )

        report = build_data_quality_report(frame)

        date_summary = report["dates"]
        codes = issue_codes(report)

        self.assertEqual(
            report["overall_status"],
            "passed_with_warnings",
        )
        self.assertEqual(
            date_summary["invalid_date_counts"][
                "termination_date"
            ],
            1,
        )
        self.assertEqual(
            date_summary["negative_reporting_lag_count"],
            1,
        )
        self.assertIn(
            "invalid_termination_date",
            codes,
        )
        self.assertIn(
            "negative_reporting_lag",
            codes,
        )

    def test_extraction_count_mismatch_is_warning(self) -> None:
        """Extraction and transformation counts should be reconciled."""

        frame = transform([make_recall()])

        report = build_data_quality_report(
            frame,
            extraction_metadata={"records_extracted": 3},
        )

        reconciliation = report["reconciliation"]

        self.assertEqual(
            report["overall_status"],
            "passed_with_warnings",
        )
        self.assertEqual(
            reconciliation["records_extracted"],
            3,
        )
        self.assertEqual(
            reconciliation["records_transformed"],
            1,
        )
        self.assertFalse(
            reconciliation["row_count_matches_extraction"]
        )
        self.assertIn(
            "extraction_transformation_count_mismatch",
            issue_codes(report),
        )

    def test_api_reported_count_mismatch_is_warning(self) -> None:
        """A changing API result set should be surfaced as a warning."""

        frame = transform([make_recall()])

        report = build_data_quality_report(
            frame,
            extraction_metadata={
                "records_reported_by_api": 3,
                "records_extracted": 1,
                "count_mismatch": True,
                "count_difference": -2,
            },
        )

        reconciliation = report["reconciliation"]

        self.assertEqual(
            report["overall_status"],
            "passed_with_warnings",
        )
        self.assertEqual(
            reconciliation["records_reported_by_api"],
            3,
        )
        self.assertEqual(
            reconciliation["records_extracted"],
            1,
        )
        self.assertFalse(
            reconciliation[
                "api_reported_count_matches_extraction"
            ]
        )
        self.assertTrue(
            reconciliation["row_count_matches_extraction"]
        )
        self.assertIn(
            "api_reported_count_mismatch",
            issue_codes(report),
        )

    def test_invalid_metadata_type_raises_error(self) -> None:
        """Extraction metadata must be a dictionary when supplied."""

        frame = transform([make_recall()])
        invalid_metadata: Any = ["not a dictionary"]

        with self.assertRaisesRegex(
            DataQualityError,
            "dictionary",
        ):
            build_data_quality_report(
                frame,
                extraction_metadata=invalid_metadata,
            )

    def test_non_dataframe_input_raises_error(self) -> None:
        """The quality layer must reject non-DataFrame input."""

        invalid_frame: Any = [{"record_id": "record-001"}]

        with self.assertRaisesRegex(
            DataQualityError,
            "DataFrame",
        ):
            build_data_quality_report(invalid_frame)


if __name__ == "__main__":
    unittest.main()
