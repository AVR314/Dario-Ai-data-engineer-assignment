"""Local persistence helpers for raw and processed project data."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    EXTRACTION_METADATA_FILE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RAW_DATA_FILE,
)


class StorageError(RuntimeError):
    """Raised when project data cannot be saved or loaded safely."""


def ensure_data_directories() -> None:
    """Create the required local data directories when they do not exist."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _write_json_atomic(
    payload: Any,
    destination: Path,
) -> None:
    """
    Write JSON through a temporary file and replace the destination atomically.

    This prevents a partially written file from replacing a valid file if the
    process is interrupted during the write operation.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            prefix=f"{destination.stem}_",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            json.dump(
                payload,
                temporary_file,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

            temporary_path = Path(temporary_file.name)

        os.replace(
            temporary_path,
            destination,
        )

    except (OSError, TypeError, ValueError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

        raise StorageError(
            f"Unable to write JSON file safely: {destination}"
        ) from exc


def _read_json(source: Path) -> Any:
    """Read and decode a local JSON file."""

    if not source.exists():
        raise StorageError(
            f"Required cache file does not exist: {source}"
        )

    try:
        with source.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except json.JSONDecodeError as exc:
        raise StorageError(
            f"Cached JSON file is invalid or corrupted: {source}"
        ) from exc

    except OSError as exc:
        raise StorageError(
            f"Unable to read cached JSON file: {source}"
        ) from exc


def save_raw_snapshot(
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    """
    Persist a successful openFDA extraction and its metadata locally.

    The original API records are stored separately from extraction metadata,
    making the source data easy to inspect and reproduce.
    """

    if not isinstance(records, list):
        raise StorageError(
            "Raw records must be provided as a list."
        )

    if not all(
        isinstance(record, dict)
        for record in records
    ):
        raise StorageError(
            "Every raw record must be represented as a dictionary."
        )

    if not isinstance(metadata, dict):
        raise StorageError(
            "Extraction metadata must be a dictionary."
        )

    ensure_data_directories()

    metadata_to_save = {
        **metadata,
        "snapshot_saved_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "raw_snapshot_path": str(
            RAW_DATA_FILE.relative_to(
                RAW_DATA_FILE.parents[2]
            )
        ),
        "metadata_path": str(
            EXTRACTION_METADATA_FILE.relative_to(
                EXTRACTION_METADATA_FILE.parents[2]
            )
        ),
    }

    _write_json_atomic(
        records,
        RAW_DATA_FILE,
    )

    _write_json_atomic(
        metadata_to_save,
        EXTRACTION_METADATA_FILE,
    )


def load_cached_raw_snapshot(
    fallback_reason: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Load the latest raw snapshot after a live extraction failure.

    Args:
        fallback_reason:
            A readable explanation of why cached data was used.

    Returns:
        A tuple containing the cached records and updated metadata.
    """

    cached_records = _read_json(
        RAW_DATA_FILE
    )

    cached_metadata = _read_json(
        EXTRACTION_METADATA_FILE
    )

    if not isinstance(cached_records, list):
        raise StorageError(
            "Cached raw data must contain a JSON list of records."
        )

    if not all(
        isinstance(record, dict)
        for record in cached_records
    ):
        raise StorageError(
            "Cached raw data contains a record that is not a JSON object."
        )

    if not isinstance(cached_metadata, dict):
        raise StorageError(
            "Cached extraction metadata must contain a JSON object."
        )

    expected_count = cached_metadata.get(
        "records_extracted"
    )

    if expected_count is not None:
        try:
            parsed_expected_count = int(
                expected_count
            )

        except (TypeError, ValueError) as exc:
            raise StorageError(
                "Cached metadata contains an invalid "
                "records_extracted value."
            ) from exc

        if parsed_expected_count != len(cached_records):
            raise StorageError(
                "Cached raw-data count does not match "
                "extraction metadata. "
                f"Metadata={parsed_expected_count}, "
                f"actual={len(cached_records)}."
            )

    previous_warnings = cached_metadata.get(
        "warnings",
        [],
    )

    if not isinstance(previous_warnings, list):
        previous_warnings = [
            str(previous_warnings)
        ]

    fallback_warning = (
        "Live API extraction failed. The project used the latest "
        "available raw snapshot instead. "
        f"Reason: {fallback_reason}"
    )

    updated_metadata = {
        **cached_metadata,
        "used_cached_data": True,
        "cache_loaded_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "fallback_reason": fallback_reason,
        "warnings": [
            *previous_warnings,
            fallback_warning,
        ],
        "extraction_status": "cached_fallback",
        "records_extracted": len(cached_records),
    }

    return cached_records, updated_metadata


def raw_snapshot_exists() -> bool:
    """Return whether both required raw snapshot files are available."""

    return (
        RAW_DATA_FILE.is_file()
        and EXTRACTION_METADATA_FILE.is_file()
    )


def save_dataframe_csv(
    frame: pd.DataFrame,
    destination: Path,
) -> None:
    """
    Save a processed DataFrame through an atomic file replacement.

    A temporary file is written first. The destination is replaced only after
    the complete CSV has been written successfully.
    """

    if not isinstance(frame, pd.DataFrame):
        raise StorageError(
            "Processed output must be provided as a pandas DataFrame."
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".tmp",
            prefix=f"{destination.stem}_",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            frame.to_csv(
                temporary_file,
                index=False,
            )

            temporary_file.flush()
            os.fsync(
                temporary_file.fileno()
            )

            temporary_path = Path(
                temporary_file.name
            )

        os.replace(
            temporary_path,
            destination,
        )

    except (OSError, TypeError, ValueError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True
            )

        raise StorageError(
            f"Unable to save processed CSV safely: {destination}"
        ) from exc


def save_json_document(
    payload: Any,
    destination: Path,
) -> None:
    """Save a general JSON document using the atomic JSON writer."""

    _write_json_atomic(
        payload,
        destination,
    )