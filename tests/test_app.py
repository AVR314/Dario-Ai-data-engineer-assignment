"""Focused tests for pure Streamlit dashboard logic."""

import unittest

import pandas as pd

from app import (
    PLOTLY_CONFIG,
    build_business_base,
    build_firm_summary,
    build_insights,
    build_monthly_trend,
    classification_chart,
    filter_records,
    format_status,
    monthly_chart,
    normalize_firm_search,
    quality_finding_label,
    short_quality_explanation,
    validate_assets,
)


def make_dashboard_frame() -> pd.DataFrame:
    """Create a compact prepared record population for dashboard tests."""

    return pd.DataFrame(
        {
            "record_id": ["record-1", "record-2", "record-3"],
            "source_record_hash": ["hash-1", "hash-2", "hash-3"],
            "recall_number": ["D-1", "D-2", "D-3"],
            "event_id": ["event-1", "event-1", "event-2"],
            "classification": ["Class I", "Class I", "Class II"],
            "severity_label": ["High", "High", "Moderate"],
            "recalling_firm_raw": [
                "Johnson & Johnson",
                "Johnson and Johnson",
                "Example Pharma",
            ],
            "recalling_firm_normalized": [
                "JOHNSON JOHNSON",
                "JOHNSON AND JOHNSON",
                "EXAMPLE PHARMA",
            ],
            "product_description": ["Product A", "Product B", "Product C"],
            "reason_for_recall": ["Reason A", "Reason B", "Reason C"],
            "recall_initiation_date": pd.to_datetime(
                ["2024-01-01", "2024-01-10", "2024-03-01"]
            ),
            "report_date": pd.to_datetime(
                ["2024-01-05", "2024-01-20", "2024-03-10"]
            ),
            "termination_date": pd.to_datetime(
                ["2024-02-01", None, "2024-04-01"]
            ),
            "status_at_publication": ["Terminated", "Ongoing", "Completed"],
            "reporting_lag_days": [4, 10, 9],
            "has_termination_date": [True, False, True],
            "is_exact_duplicate": [False, False, False],
        }
    )


class DashboardLogicTests(unittest.TestCase):
    """Validate duplicate handling, filters and filtered aggregations."""

    def test_business_base_removes_exact_source_duplicates(self) -> None:
        """Only one row per source hash should contribute to metrics."""

        frame = make_dashboard_frame()
        duplicate = frame.iloc[[0]].copy()
        duplicate["record_id"] = "record-4"
        duplicate["is_exact_duplicate"] = True
        frame = pd.concat([frame, duplicate], ignore_index=True)

        result = build_business_base(frame)

        self.assertEqual(len(result), 3)
        self.assertEqual(result["source_record_hash"].nunique(), 3)

    def test_normalized_firm_search_matches_normalized_data(self) -> None:
        """Punctuation in user input must follow transformation rules."""

        frame = make_dashboard_frame()

        result = filter_records(
            frame,
            start_date="2024-01-01",
            end_date="2024-03-31",
            selected_classifications=[],
            selected_statuses=[],
            firm_query="Johnson & Johnson",
        )

        self.assertEqual(normalize_firm_search("Johnson & Johnson"), "JOHNSON JOHNSON")
        self.assertEqual(result["record_id"].tolist(), ["record-1"])

    def test_filters_apply_dates_classification_and_status(self) -> None:
        """All primary filters should combine predictably."""

        result = filter_records(
            make_dashboard_frame(),
            start_date="2024-01-01",
            end_date="2024-01-31",
            selected_classifications=["Class I"],
            selected_statuses=["Ongoing"],
            firm_query="",
        )

        self.assertEqual(result["record_id"].tolist(), ["record-2"])

    def test_monthly_trend_includes_zero_month_and_marks_partial_month(self) -> None:
        """A partial final month must not appear as a complete rolling value."""

        trend = build_monthly_trend(
            make_dashboard_frame(),
            start_date="2024-01-01",
            end_date="2024-03-15",
        )

        self.assertEqual(trend["recall_records"].tolist(), [2, 0, 1])
        self.assertEqual(trend["report_month"].tolist(), ["2024-01", "2024-02", "2024-03"])
        self.assertTrue(bool(trend.iloc[-1]["is_partial_month"]))
        self.assertTrue(pd.isna(trend.iloc[-1]["three_month_average"]))
        self.assertIn("Partial through 2024-03-15", trend.iloc[-1]["period_status"])

    def test_monthly_plotly_chart_has_contextual_hover_and_transition(self) -> None:
        """The primary trend should support useful responsive interaction."""

        trend = build_monthly_trend(
            make_dashboard_frame(),
            start_date="2024-01-01",
            end_date="2024-03-15",
        )

        figure = monthly_chart(trend)

        self.assertEqual(
            [trace.name for trace in figure.data],
            ["Recall records", "3-month average", "Partial month"],
        )
        self.assertEqual(figure.layout.hovermode, "x unified")
        self.assertEqual(figure.layout.transition.duration, 280)
        self.assertIn("Coverage", figure.data[0].hovertemplate)
        self.assertIn("Records so far", figure.data[2].hovertemplate)

    def test_plotly_modebar_is_minimal_and_hover_driven(self) -> None:
        """Useful chart controls should remain without a persistent toolbar."""

        removed = set(PLOTLY_CONFIG["modeBarButtonsToRemove"])

        self.assertEqual(PLOTLY_CONFIG["displayModeBar"], "hover")
        self.assertFalse(PLOTLY_CONFIG["displaylogo"])
        self.assertTrue({"select2d", "lasso2d", "zoomIn2d", "zoomOut2d"} <= removed)
        self.assertTrue(
            {"zoom2d", "pan2d", "resetScale2d", "toImage"}.isdisjoint(removed)
        )

    def test_classification_plotly_chart_has_compact_hover_and_value_space(self) -> None:
        """Classification hover stays concise and direct values have room."""

        summary = pd.DataFrame(
            {
                "classification": ["Class I", "Class II"],
                "display_label": ["Class I · High", "Class II · Moderate"],
                "severity_label": ["High", "Moderate"],
                "recall_records": [4, 6],
                "share_pct": [40.0, 60.0],
                "classification_description": [
                    "Reasonable probability of serious consequences.",
                    "Temporary or medically reversible consequences.",
                ],
            }
        )

        figure = classification_chart(summary)

        self.assertEqual(len(figure.data), 1)
        self.assertEqual(figure.data[0].orientation, "h")
        self.assertEqual(figure.layout.hovermode, "closest")
        self.assertEqual(figure.layout.transition.duration, 280)
        self.assertIn("Severity", figure.data[0].hovertemplate)
        self.assertIn("Records", figure.data[0].hovertemplate)
        self.assertIn("Share", figure.data[0].hovertemplate)
        self.assertNotIn("Definition", figure.data[0].hovertemplate)
        self.assertGreater(figure.layout.xaxis.range[1], summary["recall_records"].max())
        self.assertEqual(figure.layout.margin.r, 48)

    def test_insights_are_structured_as_primary_and_supporting_stats(self) -> None:
        """Overview insight content should remain concise and data-derived."""

        frame = make_dashboard_frame()
        trend = build_monthly_trend(frame, "2024-01-01", "2024-03-31")

        insights = build_insights(frame, build_firm_summary(frame), trend)

        self.assertEqual(insights["primary"]["label"], "Leading classification")
        self.assertIn("Class I", insights["primary"]["value"])
        self.assertEqual(insights["supporting"][0]["label"], "Top-five firm share")
        self.assertIn("not company risk", insights["supporting"][0]["detail"])

    def test_quality_explanation_is_compact_without_losing_source_detail(self) -> None:
        """The default quality table should use a concise business summary."""

        issue = {
            "code": "termination_date_contextual_missingness",
            "message": "A much longer technical explanation remains available elsewhere.",
        }

        result = short_quality_explanation(issue)

        self.assertEqual(
            result,
            "Missing termination dates are contextual, not quality failures.",
        )
        self.assertEqual(
            quality_finding_label("termination_date_contextual_missingness"),
            "Termination-date coverage",
        )

    def test_firm_summary_keeps_record_and_event_counts_distinct(self) -> None:
        """Firm volume must retain unique-event context."""

        summary = build_firm_summary(make_dashboard_frame())
        johnson = summary[summary["firm"].eq("JOHNSON JOHNSON")].iloc[0]

        self.assertEqual(johnson["recall_records"], 1)
        self.assertEqual(johnson["unique_events"], 1)
        self.assertEqual(johnson["class_i_records"], 1)
        self.assertEqual(johnson["class_i_share_pct"], 100.0)

    def test_machine_status_is_human_readable(self) -> None:
        """Quality statuses should be understandable to business users."""

        self.assertEqual(format_status("passed_with_warnings"), "Passed With Warnings")
        self.assertEqual(format_status("cached_fallback"), "Cached Fallback")
        self.assertEqual(format_status(None), "Not available")

    def test_empty_filtered_result_is_supported(self) -> None:
        """An unmatched filter should return a valid empty DataFrame."""

        result = filter_records(
            make_dashboard_frame(),
            start_date="2024-01-01",
            end_date="2024-03-31",
            selected_classifications=[],
            selected_statuses=[],
            firm_query="No Such Firm",
        )

        self.assertTrue(result.empty)
        self.assertTrue(build_firm_summary(result).empty)

    def test_asset_validation_reports_missing_schema(self) -> None:
        """Incompatible processed files should produce actionable errors."""

        errors = validate_assets(
            recalls=pd.DataFrame({"record_id": ["record-1"]}),
            monthly=pd.DataFrame(),
            firms=pd.DataFrame(),
            classification=pd.DataFrame(),
            quality_report={},
            extraction_metadata={},
        )

        self.assertTrue(errors)
        self.assertTrue(any("recalls_enriched.csv" in error for error in errors))
        self.assertTrue(any("data_quality_report.json" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
