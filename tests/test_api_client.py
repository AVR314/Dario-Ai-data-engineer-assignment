"""Tests for the openFDA extraction client."""

import unittest
from unittest.mock import Mock, patch

import requests

from src import api_client


class FakeResponse:
    """Minimal HTTP response object for deterministic tests."""

    def __init__(
        self,
        status_code: int,
        payload=None,
        headers=None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text

    def json(self):
        """Return the configured payload or raise a configured exception."""

        if isinstance(self._payload, Exception):
            raise self._payload

        return self._payload


def make_payload(records: list[dict], total: int) -> dict:
    """Create a valid openFDA-style response payload."""

    return {
        "meta": {
            "results": {
                "skip": 0,
                "limit": len(records),
                "total": total,
            }
        },
        "results": records,
    }


class APIClientTests(unittest.TestCase):
    """Validate query construction, retries and pagination behavior."""

    def test_build_search_query_with_valid_dates(self) -> None:
        """Valid dates should produce the expected openFDA query."""

        query = api_client.build_search_query(
            "20210101",
            "20211231",
        )

        self.assertEqual(
            query,
            "report_date:[20210101 TO 20211231]",
        )

    def test_build_search_query_rejects_invalid_format(self) -> None:
        """Dates must use the compact YYYYMMDD format."""

        with self.assertRaisesRegex(
            ValueError,
            "YYYYMMDD",
        ):
            api_client.build_search_query(
                "2021-01-01",
                "20211231",
            )

    def test_build_search_query_rejects_reversed_range(self) -> None:
        """The start date must not be later than the end date."""

        with self.assertRaisesRegex(
            ValueError,
            "must not be later",
        ):
            api_client.build_search_query(
                "20220101",
                "20210101",
            )

    def test_invalid_dates_fail_before_session_creation(self) -> None:
        """Invalid dates must fail before any network resources are created."""

        with patch.object(api_client.requests, "Session") as session_mock:
            with self.assertRaises(ValueError):
                api_client.fetch_drug_recalls(
                    "2021-01-01",
                    "20211231",
                )

        session_mock.assert_not_called()

    def test_not_found_response_returns_empty_results(self) -> None:
        """openFDA NOT_FOUND should represent a valid empty dataset."""

        session = Mock()
        session.get.return_value = FakeResponse(
            status_code=404,
            payload={
                "error": {
                    "code": "NOT_FOUND",
                    "message": "No matches found!",
                }
            },
        )

        payload = api_client._request_page(
            session,
            search_query="report_date:[20990101 TO 20991231]",
            skip=0,
            limit=1000,
        )

        self.assertEqual(payload["results"], [])
        self.assertEqual(
            payload["meta"]["results"]["total"],
            0,
        )

    def test_other_404_response_remains_an_error(self) -> None:
        """A non-NOT_FOUND 404 must not be converted into empty data."""

        session = Mock()
        session.get.return_value = FakeResponse(
            status_code=404,
            payload={
                "error": {
                    "code": "UNKNOWN_ENDPOINT",
                    "message": "Unknown endpoint",
                }
            },
        )

        with self.assertRaisesRegex(
            api_client.OpenFDAClientError,
            "UNKNOWN_ENDPOINT",
        ):
            api_client._request_page(
                session,
                search_query="test",
                skip=0,
                limit=1000,
            )

    @patch.object(api_client.time, "sleep")
    def test_retry_after_header_is_respected(
        self,
        sleep_mock: Mock,
    ) -> None:
        """HTTP Retry-After should control the retry delay."""

        session = Mock()
        session.get.side_effect = [
            FakeResponse(
                status_code=429,
                payload={
                    "error": {
                        "code": "RATE_LIMIT",
                        "message": "Too many requests",
                    }
                },
                headers={"Retry-After": "7"},
            ),
            FakeResponse(
                status_code=200,
                payload=make_payload(
                    [{"recall_number": "D-1001"}],
                    total=1,
                ),
            ),
        ]

        payload = api_client._request_page(
            session,
            search_query="test",
            skip=0,
            limit=1000,
        )

        self.assertEqual(len(payload["results"]), 1)
        sleep_mock.assert_called_once_with(7)
        self.assertEqual(session.get.call_count, 2)

    @patch.object(api_client.time, "sleep")
    def test_final_failure_reports_latest_network_error(
        self,
        sleep_mock: Mock,
    ) -> None:
        """Mixed retry failures must report the actual final failure."""

        session = Mock()
        session.get.side_effect = [
            requests.ConnectionError("first network failure"),
            FakeResponse(
                status_code=503,
                payload={
                    "error": {
                        "code": "SERVER_ERROR",
                        "message": "Temporary server failure",
                    }
                },
            ),
            requests.ConnectionError("final network failure"),
        ]

        with patch.object(api_client, "MAX_RETRIES", 3):
            with self.assertRaisesRegex(
                api_client.OpenFDAClientError,
                "final network failure",
            ):
                api_client._request_page(
                    session,
                    search_query="test",
                    skip=0,
                    limit=1000,
                )

        self.assertEqual(session.get.call_count, 3)
        self.assertEqual(sleep_mock.call_count, 2)

    def test_successful_exact_multiple_pagination(self) -> None:
        """An exact page-size multiple must not trigger an extra request."""

        first_page = make_payload(
            [
                {"recall_number": "D-1001"},
                {"recall_number": "D-1002"},
            ],
            total=4,
        )
        second_page = make_payload(
            [
                {"recall_number": "D-1003"},
                {"recall_number": "D-1004"},
            ],
            total=4,
        )

        session = Mock()
        session.headers = Mock()

        with (
            patch.object(api_client, "PAGE_SIZE", 2),
            patch.object(api_client.requests, "Session") as session_class,
            patch.object(
                api_client,
                "_request_page",
                side_effect=[first_page, second_page],
            ) as request_page_mock,
        ):
            session_class.return_value.__enter__.return_value = session

            records, metadata = api_client.fetch_drug_recalls(
                "20210101",
                "20210131",
            )

        self.assertEqual(len(records), 4)
        self.assertEqual(metadata["pages_requested"], 2)
        self.assertFalse(metadata["count_mismatch"])
        self.assertEqual(request_page_mock.call_count, 2)

    def test_successful_non_exact_pagination(self) -> None:
        """A partial final page should end pagination cleanly."""

        pages = [
            make_payload(
                [
                    {"recall_number": "D-1001"},
                    {"recall_number": "D-1002"},
                ],
                total=5,
            ),
            make_payload(
                [
                    {"recall_number": "D-1003"},
                    {"recall_number": "D-1004"},
                ],
                total=5,
            ),
            make_payload(
                [{"recall_number": "D-1005"}],
                total=5,
            ),
        ]

        session = Mock()
        session.headers = Mock()

        with (
            patch.object(api_client, "PAGE_SIZE", 2),
            patch.object(api_client.requests, "Session") as session_class,
            patch.object(
                api_client,
                "_request_page",
                side_effect=pages,
            ) as request_page_mock,
        ):
            session_class.return_value.__enter__.return_value = session

            records, metadata = api_client.fetch_drug_recalls(
                "20210101",
                "20210131",
            )

        self.assertEqual(len(records), 5)
        self.assertEqual(metadata["pages_requested"], 3)
        self.assertFalse(metadata["count_mismatch"])
        self.assertEqual(request_page_mock.call_count, 3)

    def test_count_mismatch_is_recorded_as_warning(self) -> None:
        """A total-count mismatch should warn without failing extraction."""

        incomplete_page = make_payload(
            [
                {"recall_number": "D-1001"},
                {"recall_number": "D-1002"},
            ],
            total=5,
        )

        session = Mock()
        session.headers = Mock()

        with (
            patch.object(api_client, "PAGE_SIZE", 3),
            patch.object(api_client.requests, "Session") as session_class,
            patch.object(
                api_client,
                "_request_page",
                return_value=incomplete_page,
            ),
        ):
            session_class.return_value.__enter__.return_value = session

            records, metadata = api_client.fetch_drug_recalls(
                "20210101",
                "20210131",
            )

        self.assertEqual(len(records), 2)
        self.assertTrue(metadata["count_mismatch"])
        self.assertEqual(metadata["count_difference"], -3)
        self.assertEqual(
            metadata["extraction_status"],
            "success_with_warnings",
        )
        self.assertTrue(metadata["warnings"])

    def test_success_response_without_results_fails(self) -> None:
        """A malformed success payload must not be accepted silently."""

        session = Mock()
        session.get.return_value = FakeResponse(
            status_code=200,
            payload={
                "meta": {
                    "results": {
                        "total": 1,
                    }
                }
            },
        )

        with self.assertRaisesRegex(
            api_client.OpenFDAClientError,
            "does not contain a 'results' field",
        ):
            api_client._request_page(
                session,
                search_query="test",
                skip=0,
                limit=1000,
            )

    def test_query_above_pagination_limit_fails(self) -> None:
        """Queries larger than the supported range must fail clearly."""

        oversized_payload = make_payload(
            [{"recall_number": "D-1001"}],
            total=5,
        )

        session = Mock()
        session.headers = Mock()

        with (
            patch.object(api_client, "PAGE_SIZE", 2),
            patch.object(api_client, "MAX_API_SKIP", 2),
            patch.object(api_client.requests, "Session") as session_class,
            patch.object(
                api_client,
                "_request_page",
                return_value=oversized_payload,
            ),
        ):
            session_class.return_value.__enter__.return_value = session

            with self.assertRaisesRegex(
                api_client.OpenFDAClientError,
                "exceeds the supported pagination range",
            ):
                api_client.fetch_drug_recalls(
                    "20210101",
                    "20210131",
                )


if __name__ == "__main__":
    unittest.main()