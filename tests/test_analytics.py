"""Tests for business analytics output tables."""

import unittest
from typing import Any

import pandas as pd

from src.analytics import (
    AnalyticsError,
    build_analytics_outputs,
    build_firm_summary,
    build_monthly_summary,
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


class AnalyticsTests(unittest.TestCase):
    """Validate monthly and company-level analytical outputs."""

    def test_missing_month_is_inserted_with_zero_recalls(self) -> None:
        """A month without source records must remain visible as zero."""

        frame = transform(
            [
                make_recall(
                    recall_number="D-1001-2024",
                    event_id="90001",
                    report_date="20240110",
                ),
                make_recall(
                    recall_number="D-1002-2024",
                    event_id="90002",
                    report_date="20240310",
                ),
            ]
        )

        summary = build_monthly_summary(
            frame,
            start_date="20240101",
            end_date="20240331",
        )

        self.assertEqual(
            summary["report_month"].tolist(),
            ["2024-01", "2024-02", "2024-03"],
        )

        self.assertEqual(
            summary["total_recalls"].tolist(),
            [1, 0, 1],
        )

    def test_rolling_average_uses_continuous_months(self) -> None:
        """The rolling average must include months with zero recalls."""

        frame = transform(
            [
                make_recall(
                    recall_number="D-1001-2024",
                    report_date="20240110",
                ),
                make_recall(
                    recall_number="D-1002-2024",
                    event_id="90002",
                    report_date="20240310",
                ),
            ]
        )

        summary = build_monthly_summary(
            frame,
            start_date="20240101",
            end_date="20240331",
        )

        rolling_values = summary[
            "three_month_rolling_avg_recalls"
        ].tolist()

        self.assertEqual(rolling_values[0], 1.0)
        self.assertEqual(rolling_values[1], 0.5)
        self.assertAlmostEqual(
            rolling_values[2],
            0.67,
            places=2,
        )

    def test_classification_counts_are_calculated_correctly(
        self,
    ) -> None:
        """Monthly totals should be split correctly by classification."""

        frame = transform(
            [
                make_recall(
                    recall_number="D-1001-2024",
                    event_id="90001",
                    classification="Class I",
                    report_date="20240110",
                ),
                make_recall(
                    recall_number="D-1002-2024",
                    event_id="90002",
                    classification="Class II",
                    report_date="20240115",
                ),
                make_recall(
                    recall_number="D-1003-2024",
                    event_id="90003",
                    classification="Class III",
                    report_date="20240120",
                ),
            ]
        )

        summary = build_monthly_summary(
            frame,
            start_date="20240101",
            end_date="20240131",
        )

        row = summary.iloc[0]

        self.assertEqual(row["total_recalls"], 3)
        self.assertEqual(row["class_i_recalls"], 1)
        self.assertEqual(row["class_ii_recalls"], 1)
        self.assertEqual(row["class_iii_recalls"], 1)
        self.assertEqual(
            row["not_yet_classified_recalls"],
            0,
        )

    def test_exact_duplicates_do_not_inflate_business_metrics(
        self,
    ) -> None:
        """Exact source duplicates should count once in analytics."""

        duplicated_record = make_recall()

        frame = transform(
            [
                duplicated_record,
                duplicated_record.copy(),
            ]
        )

        monthly_summary, firm_summary = build_analytics_outputs(
            frame,
            start_date="20240101",
            end_date="20240131",
        )

        self.assertEqual(
            monthly_summary.iloc[0]["total_recalls"],
            1,
        )

        self.assertEqual(
            firm_summary.iloc[0]["recall_count"],
            1,
        )

    def test_firm_summary_calculates_business_metrics(self) -> None:
        """Company-level metrics should aggregate normalized firms."""

        frame = transform(
            [
                make_recall(
                    recall_number="D-1001-2023",
                    event_id="90001",
                    classification="Class I",
                    recalling_firm="Example Pharma, Inc.",
                    recall_initiation_date="20230101",
                    report_date="20230111",
                    termination_date=None,
                ),
                make_recall(
                    recall_number="D-1002-2024",
                    event_id="90002",
                    classification="Class II",
                    recalling_firm="EXAMPLE PHARMA INC",
                    recall_initiation_date="20240101",
                    report_date="20240121",
                    termination_date="20240201",
                ),
            ]
        )

        summary = build_firm_summary(frame)

        self.assertEqual(len(summary), 1)

        row = summary.iloc[0]

        self.assertEqual(
            row["recalling_firm_normalized"],
            "EXAMPLE PHARMA INC",
        )
        self.assertEqual(row["recall_count"], 2)
        self.assertEqual(row["unique_event_count"], 2)
        self.assertEqual(row["class_i_count"], 1)
        self.assertEqual(row["class_i_pct"], 50.0)
        self.assertEqual(row["active_year_count"], 2)
        self.assertEqual(
            row["median_reporting_lag_days"],
            15.0,
        )

    def test_termination_coverage_is_calculated_per_month(
        self,
    ) -> None:
        """Termination coverage should reflect valid available dates."""

        frame = transform(
            [
                make_recall(
                    recall_number="D-1001-2024",
                    event_id="90001",
                    report_date="20240110",
                    termination_date="20240201",
                ),
                make_recall(
                    recall_number="D-1002-2024",
                    event_id="90002",
                    report_date="20240115",
                    termination_date=None,
                ),
            ]
        )

        summary = build_monthly_summary(
            frame,
            start_date="20240101",
            end_date="20240131",
        )

        self.assertEqual(
            summary.iloc[0][
                "termination_date_coverage_pct"
            ],
            50.0,
        )

    def test_empty_dataset_returns_complete_month_range(self) -> None:
        """An empty dataset should still return continuous zero months."""

        empty_frame, _ = transform_recalls(
            [],
            source_extract_timestamp="2026-08-06T12:00:00+00:00",
        )

        monthly_summary, firm_summary = build_analytics_outputs(
            empty_frame,
            start_date="20240101",
            end_date="20240331",
        )

        self.assertEqual(len(monthly_summary), 3)
        self.assertEqual(
            monthly_summary["total_recalls"].tolist(),
            [0, 0, 0],
        )
        self.assertTrue(firm_summary.empty)

    def test_reversed_date_range_raises_error(self) -> None:
        """Analytics date boundaries must be logically ordered."""

        frame = transform([make_recall()])

        with self.assertRaisesRegex(
            AnalyticsError,
            "must not be later",
        ):
            build_monthly_summary(
                frame,
                start_date="20241231",
                end_date="20240101",
            )

    def test_missing_analytics_columns_raise_error(self) -> None:
        """An incomplete transformed table must fail clearly."""

        incomplete_frame = pd.DataFrame(
            {
                "record_id": ["record-001"],
            }
        )

        with self.assertRaisesRegex(
            AnalyticsError,
            "missing analytics columns",
        ):
            build_firm_summary(incomplete_frame)


if __name__ == "__main__":
    unittest.main()