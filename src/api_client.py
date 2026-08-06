"""Client for extracting drug recall records from the openFDA API."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import requests

from src.config import (
    API_BASE_URL,
    END_DATE,
    MAX_API_SKIP,
    MAX_RETRIES,
    PAGE_SIZE,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_BACKOFF_SECONDS,
    START_DATE,
)

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class OpenFDAClientError(RuntimeError):
    """Raised when data cannot be retrieved safely from openFDA."""


def _parse_api_date(value: str, field_name: str) -> datetime:
    """Parse an openFDA date and raise a clear error for invalid values."""

    try:
        return datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must use YYYYMMDD format. Received: {value!r}"
        ) from exc


def build_search_query(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> str:
    """Build and validate an openFDA report-date range query."""

    parsed_start_date = _parse_api_date(start_date, "start_date")
    parsed_end_date = _parse_api_date(end_date, "end_date")

    if parsed_start_date > parsed_end_date:
        raise ValueError(
            "start_date must not be later than end_date. "
            f"Received start_date={start_date!r}, end_date={end_date!r}."
        )

    return f"report_date:[{start_date} TO {end_date}]"


def _extract_api_error(
    response: requests.Response,
) -> tuple[str | None, str]:
    """Return the structured openFDA error code and readable message."""

    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError:
        message = response.text.strip() or "No error details were provided."
        return None, message

    if not isinstance(payload, dict):
        return None, "The API returned an unexpected error response."

    error = payload.get("error")

    if isinstance(error, dict):
        error_code = error.get("code")
        error_message = (
            error.get("message")
            or error_code
            or "No error details were provided."
        )

        return (
            str(error_code) if error_code is not None else None,
            str(error_message),
        )

    if error is not None:
        return None, str(error)

    return None, "No error details were provided."


def _get_retry_wait_seconds(
    response: requests.Response,
    attempt: int,
) -> int:
    """Calculate the wait time before retrying a failed request."""

    retry_after = response.headers.get("Retry-After")

    if retry_after and retry_after.isdigit():
        return int(retry_after)

    # A simple linear backoff is sufficient for this small local project.
    return RETRY_BACKOFF_SECONDS * attempt


def _extract_total_available(payload: dict[str, Any]) -> int:
    """Read and validate the total result count from openFDA metadata."""

    meta = payload.get("meta")

    if not isinstance(meta, dict):
        raise OpenFDAClientError(
            "openFDA response does not contain valid 'meta' information."
        )

    results_meta = meta.get("results")

    if not isinstance(results_meta, dict):
        raise OpenFDAClientError(
            "openFDA response does not contain valid result metadata."
        )

    total = results_meta.get("total")

    try:
        parsed_total = int(total)
    except (TypeError, ValueError) as exc:
        raise OpenFDAClientError(
            "openFDA response does not contain a valid total result count."
        ) from exc

    if parsed_total < 0:
        raise OpenFDAClientError(
            "openFDA returned a negative total result count."
        )

    return parsed_total


def _request_page(
    session: requests.Session,
    *,
    search_query: str,
    skip: int,
    limit: int,
) -> dict[str, Any]:
    """Request one page from openFDA with bounded retries."""

    params: dict[str, Any] = {
        "search": search_query,
        "sort": "report_date:asc",
        "limit": limit,
        "skip": skip,
    }

    api_key = os.getenv("OPENFDA_API_KEY")

    if api_key:
        params["api_key"] = api_key

    last_network_error: Exception | None = None
    last_http_status: int | None = None
    last_api_error_code: str | None = None
    last_api_error_message: str | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                API_BASE_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

        except (requests.Timeout, requests.ConnectionError) as exc:
            # The latest failure was a network failure, so previous HTTP
            # failure information must not be reported as the final failure.
            last_network_error = exc
            last_http_status = None
            last_api_error_code = None
            last_api_error_message = None

            if attempt < MAX_RETRIES:
                wait_seconds = RETRY_BACKOFF_SECONDS * attempt

                logger.warning(
                    "Network error while calling openFDA. "
                    "Retrying in %s seconds, attempt %s of %s: %s",
                    wait_seconds,
                    attempt,
                    MAX_RETRIES,
                    exc,
                )

                time.sleep(wait_seconds)
                continue

            break

        if response.status_code == 200:
            try:
                payload = response.json()
            except requests.exceptions.JSONDecodeError as exc:
                raise OpenFDAClientError(
                    "openFDA returned a response that was not valid JSON."
                ) from exc

            if not isinstance(payload, dict):
                raise OpenFDAClientError(
                    "openFDA returned an unexpected non-object response."
                )

            if "results" not in payload:
                raise OpenFDAClientError(
                    "openFDA response does not contain a 'results' field."
                )

            if not isinstance(payload["results"], list):
                raise OpenFDAClientError(
                    "openFDA returned an unexpected 'results' structure."
                )

            return payload

        error_code, error_message = _extract_api_error(response)

        # openFDA represents a valid zero-result search as HTTP 404.
        if response.status_code == 404 and error_code == "NOT_FOUND":
            return {
                "meta": {
                    "results": {
                        "skip": skip,
                        "limit": limit,
                        "total": 0,
                    }
                },
                "results": [],
            }

        if response.status_code in RETRYABLE_STATUS_CODES:
            # The latest failure was an HTTP failure, so a previous network
            # error must not be reported as the final failure.
            last_network_error = None
            last_http_status = response.status_code
            last_api_error_code = error_code
            last_api_error_message = error_message

            if attempt < MAX_RETRIES:
                wait_seconds = _get_retry_wait_seconds(response, attempt)

                logger.warning(
                    "openFDA returned HTTP %s. "
                    "Retrying in %s seconds, attempt %s of %s. "
                    "API error: %s",
                    response.status_code,
                    wait_seconds,
                    attempt,
                    MAX_RETRIES,
                    error_message,
                )

                time.sleep(wait_seconds)
                continue

            break

        raise OpenFDAClientError(
            f"openFDA request failed with HTTP {response.status_code}. "
            f"API error code: {error_code or 'unknown'}. "
            f"Message: {error_message}"
        )

    if last_http_status is not None:
        raise OpenFDAClientError(
            "Unable to retrieve data from openFDA after "
            f"{MAX_RETRIES} attempts. "
            f"Last HTTP status: {last_http_status}. "
            f"Last API error code: {last_api_error_code or 'unknown'}. "
            f"Last message: "
            f"{last_api_error_message or 'No error details were provided.'}"
        )

    raise OpenFDAClientError(
        "Unable to retrieve data from openFDA after "
        f"{MAX_RETRIES} attempts because of a network error. "
        f"Last error: {last_network_error or 'unknown network error'}"
    ) from last_network_error


def fetch_drug_recalls(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Fetch all available drug recall records for the selected date range.

    Returns:
        A tuple containing the extracted records and extraction metadata.

    Raises:
        ValueError:
            If the supplied dates are invalid.
        OpenFDAClientError:
            If extraction fails or exceeds API pagination limits.
    """

    # Validation happens before creating a session or sending a request.
    search_query = build_search_query(start_date, end_date)

    records: list[dict[str, Any]] = []
    total_available: int | None = None
    page_number = 0

    extraction_started_at = datetime.now(timezone.utc)

    with requests.Session() as session:
        session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "Dario-AI-Data-Engineer-Assignment/1.0",
            }
        )

        while True:
            skip = len(records)

            if skip > MAX_API_SKIP:
                raise OpenFDAClientError(
                    "The query exceeded openFDA's supported pagination range. "
                    "Use a smaller date range."
                )

            page_number += 1

            logger.info(
                "Fetching openFDA page %s with skip=%s and limit=%s.",
                page_number,
                skip,
                PAGE_SIZE,
            )

            payload = _request_page(
                session,
                search_query=search_query,
                skip=skip,
                limit=PAGE_SIZE,
            )

            page_records = payload["results"]

            if total_available is None:
                total_available = _extract_total_available(payload)

                logger.info(
                    "openFDA reports %s matching records.",
                    total_available,
                )

                maximum_supported_results = MAX_API_SKIP + PAGE_SIZE

                if total_available > maximum_supported_results:
                    raise OpenFDAClientError(
                        f"The query matched {total_available} records, which "
                        f"exceeds the supported pagination range of "
                        f"{maximum_supported_results} records. "
                        "Use a smaller date range."
                    )

            records.extend(page_records)

            logger.info(
                "Fetched %s of %s reported records.",
                len(records),
                total_available,
            )

            if not page_records:
                break

            if len(page_records) < PAGE_SIZE:
                break

            if len(records) >= total_available:
                break

    extraction_finished_at = datetime.now(timezone.utc)

    count_mismatch = (
        total_available is not None
        and len(records) != total_available
    )

    warnings: list[str] = []

    if count_mismatch:
        warning_message = (
            "The number of extracted records does not match the total "
            f"reported by openFDA. Reported={total_available}, "
            f"extracted={len(records)}."
        )

        warnings.append(warning_message)
        logger.warning(warning_message)

    metadata = {
        "source": "openFDA drug enforcement API",
        "endpoint": API_BASE_URL,
        "search_query": search_query,
        "start_date": start_date,
        "end_date": end_date,
        "records_reported_by_api": total_available,
        "records_extracted": len(records),
        "count_mismatch": count_mismatch,
        "count_difference": (
            len(records) - total_available
            if total_available is not None
            else None
        ),
        "pages_requested": page_number,
        "page_size": PAGE_SIZE,
        "api_key_used": bool(os.getenv("OPENFDA_API_KEY")),
        "extraction_started_at_utc": extraction_started_at.isoformat(),
        "extraction_finished_at_utc": extraction_finished_at.isoformat(),
        "used_cached_data": False,
        "warnings": warnings,
        "extraction_status": (
            "success_with_warnings"
            if warnings
            else "success"
        ),
    }

    return records, metadata