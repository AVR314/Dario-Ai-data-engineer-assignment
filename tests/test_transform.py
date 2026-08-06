"""Tests for drug recall transformation logic."""

import unittest
from typing import Any

import pandas as pd

from src.transform import (
    TransformationError,
    normalize_firm_name,
    transform_recalls,
)


def make_recall(**overrides: Any) -> dict[str, Any]:
    """Create a valid sample recall record with optional overrides."""

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


class TransformTests(unittest.TestCase):
    """Validate cleaning, identifiers, joins and derived metrics."""

    def test_normalize_firm_name_conservatively(self) -> None:
        """Punctuation and casing should be standardized safely."""

        normalized = normalize_firm_name(
            "  Example Pharma, Inc.  "
        )

        self.assertEqual(
            normalized,
            "EXAMPLE PHARMA INC",
        )

    def test_missing_text_markers_are_normalized_safely(self) -> None:
        """Known placeholders should become null only on an exact match."""

        transformed, _ = transform_recalls(
            [
                make_recall(
                    recall_number=" N/A ",
                    voluntary_mandated="not available",
                    state="NULL",
                    product_description=(
                        "The text N/A appears as part of this description"
                    ),
                )
            ],
            source_extract_timestamp="2026-08-06T12:00:00+00:00",
        )

        row = transformed.iloc[0]

        self.assertTrue(
            pd.isna(row["recall_number"])
        )

        self.assertTrue(
            pd.isna(row["voluntary_mandated"])
        )

        self.assertTrue(
            pd.isna(row["state"])
        )

        self.assertEqual(
            row["product_description"],
            "The text N/A appears as part of this description",
        )

    def test_transform_adds_classification_information(self) -> None:
        """Classification details should be joined many-to-one."""

        transformed, dimension = transform_recalls(
            [make_recall()],
            source_extract_timestamp="2026-08-06T12:00:00+00:00",
        )

        self.assertEqual(len(transformed), 1)
        self.assertEqual(len(dimension), 4)

        row = transformed.iloc[0]

        self.assertEqual(
            row["classification"],
            "Class II",
        )
        self.assertEqual(
            row["severity_rank"],
            2,
        )
        self.assertEqual(
            row["severity_label"],
            "Moderate",
        )
        self.assertIsInstance(
            row["classification_description"],
            str,
        )

    def test_date_metrics_are_calculated_correctly(self) -> None:
        """Valid dates should produce expected historical durations."""

        transformed, _ = transform_recalls(
            [make_recall()],
            source_extract_timestamp="2026-08-06T12:00:00+00:00",
        )

        row = transformed.iloc[0]

        self.assertEqual(
            row["reporting_lag_days"],
            9,
        )
        self.assertEqual(
            row["termination_days"],
            31,
        )
        self.assertTrue(
            row["has_termination_date"]
        )
        self.assertEqual(
            row["report_month"],
            "2024-01",
        )
        self.assertFalse(
            row["negative_reporting_lag_flag"]
        )
        self.assertFalse(
            row["negative_termination_duration_flag"]
        )

    def test_negative_date_durations_are_excluded(self) -> None:
        """Negative durations must become missing and be flagged."""

        record = make_recall(
            recall_initiation_date="20240210",
            report_date="20240201",
            termination_date="20240120",
        )

        transformed, _ = transform_recalls(
            [record],
            source_extract_timestamp="2026-08-06T12:00:00+00:00",
        )

        row = transformed.iloc[0]

        self.assertTrue(
            row["negative_reporting_lag_flag"]
        )
        self.assertTrue(
            row["negative_termination_duration_flag"]
        )
        self.assertTrue(
            pd.isna(row["reporting_lag_days"])
        )
        self.assertTrue(
            pd.isna(row["termination_days"])
        )

    def test_invalid_date_is_flagged_without_crashing(self) -> None:
        """A malformed source date should be preserved as a quality flag."""

        record = make_recall(
            termination_date="not-a-date",
        )

        transformed, _ = transform_recalls(
            [record],
            source_extract_timestamp="2026-08-06T12:00:00+00:00",
        )

        row = transformed.iloc[0]

        self.assertTrue(
            row["invalid_termination_date_flag"]
        )
        self.assertTrue(
            pd.isna(row["termination_date"])
        )
        self.assertFalse(
            row["has_termination_date"]
        )
        self.assertTrue(
            pd.isna(row["termination_days"])
        )

    def test_exact_duplicates_are_preserved_and_flagged(self) -> None:
        """Exact source duplicates should remain visible for quality review."""

        duplicated_record = make_recall()

        transformed, _ = transform_recalls(
            [
                duplicated_record,
                duplicated_record.copy(),
            ],
            source_extract_timestamp="2026-08-06T12:00:00+00:00",
        )

        self.assertEqual(
            len(transformed),
            2,
        )
        self.assertEqual(
            transformed["source_record_hash"].nunique(),
            1,
        )
        self.assertEqual(
            transformed["record_id"].nunique(),
            2,
        )
        self.assertTrue(
            transformed["is_exact_duplicate"].all()
        )
        self.assertTrue(
            transformed["has_repeated_recall_number"].all()
        )
        self.assertTrue(
            (
                transformed["recall_number_occurrence_count"]
                == 2
            ).all()
        )

    def test_repeated_recall_number_is_not_treated_as_exact_duplicate(
        self,
    ) -> None:
        """Different rows sharing a recall number should not be deleted."""

        records = [
            make_recall(
                product_description="Product A",
            ),
            make_recall(
                product_description="Product B",
                event_id="90002",
            ),
        ]

        transformed, _ = transform_recalls(
            records,
            source_extract_timestamp="2026-08-06T12:00:00+00:00",
        )

        self.assertEqual(
            len(transformed),
            2,
        )
        self.assertEqual(
            transformed["source_record_hash"].nunique(),
            2,
        )
        self.assertFalse(
            transformed["is_exact_duplicate"].any()
        )
        self.assertTrue(
            transformed["has_repeated_recall_number"].all()
        )

    def test_missing_optional_fields_are_supported(self) -> None:
        """Missing optional API fields should become null values."""

        record = {
            "recall_number": "D-2001-2024",
            "classification": "Class III",
            "report_date": "20240301",
        }

        transformed, _ = transform_recalls(
            [record],
            source_extract_timestamp="2026-08-06T12:00:00+00:00",
        )

        row = transformed.iloc[0]

        self.assertEqual(
            row["recall_number"],
            "D-2001-2024",
        )
        self.assertTrue(
            pd.isna(row["termination_date"])
        )
        self.assertTrue(
            pd.isna(row["recalling_firm_raw"])
        )
        self.assertFalse(
            row["has_termination_date"]
        )

    def test_unknown_classification_remains_visible(self) -> None:
        """Unexpected categories should remain in the data for validation."""

        transformed, _ = transform_recalls(
            [
                make_recall(
                    classification="Unexpected Class",
                )
            ],
            source_extract_timestamp="2026-08-06T12:00:00+00:00",
        )

        row = transformed.iloc[0]

        self.assertEqual(
            row["classification"],
            "Unexpected Class",
        )
        self.assertTrue(
            pd.isna(row["severity_rank"])
        )
        self.assertTrue(
            pd.isna(row["severity_label"])
        )

    def test_invalid_extract_timestamp_raises_error(self) -> None:
        """An invalid extraction timestamp must fail clearly."""

        with self.assertRaisesRegex(
            TransformationError,
            "source_extract_timestamp",
        ):
            transform_recalls(
                [make_recall()],
                source_extract_timestamp="invalid timestamp",
            )

    def test_non_dictionary_record_raises_error(self) -> None:
        """Every raw record must be represented as a dictionary."""

        invalid_records: Any = [
            "not a dictionary"
        ]

        with self.assertRaisesRegex(
            TransformationError,
            "dictionary",
        ):
            transform_recalls(
                invalid_records
            )


if __name__ == "__main__":
    unittest.main()