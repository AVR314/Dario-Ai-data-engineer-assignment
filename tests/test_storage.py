"""Tests for local raw-data persistence and cache fallback."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import storage


class StorageTests(unittest.TestCase):
    """Validate snapshot saving, loading and corruption detection."""

    def setUp(self) -> None:
        """Create isolated temporary paths for every test."""

        self.temp_directory = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_directory.name)

        self.raw_directory = self.project_root / "data" / "raw"
        self.processed_directory = self.project_root / "data" / "processed"

        self.raw_file = self.raw_directory / "drug_recalls_raw.json"
        self.metadata_file = self.raw_directory / "extraction_metadata.json"

        self.path_patcher = patch.multiple(
            storage,
            RAW_DATA_DIR=self.raw_directory,
            PROCESSED_DATA_DIR=self.processed_directory,
            RAW_DATA_FILE=self.raw_file,
            EXTRACTION_METADATA_FILE=self.metadata_file,
        )
        self.path_patcher.start()

    def tearDown(self) -> None:
        """Restore patched paths and remove temporary files."""

        self.path_patcher.stop()
        self.temp_directory.cleanup()

    def test_save_and_load_cached_snapshot(self) -> None:
        """A valid snapshot should be saved and loaded as fallback data."""

        records = [
            {
                "recall_number": "D-1001",
                "classification": "Class II",
            }
        ]

        metadata = {
            "records_extracted": 1,
            "warnings": [],
            "used_cached_data": False,
            "extraction_status": "success",
        }

        storage.save_raw_snapshot(records, metadata)

        self.assertTrue(storage.raw_snapshot_exists())
        self.assertTrue(self.raw_file.is_file())
        self.assertTrue(self.metadata_file.is_file())

        loaded_records, loaded_metadata = (
            storage.load_cached_raw_snapshot("Temporary API failure")
        )

        self.assertEqual(loaded_records, records)
        self.assertTrue(loaded_metadata["used_cached_data"])
        self.assertEqual(
            loaded_metadata["extraction_status"],
            "cached_fallback",
        )
        self.assertEqual(
            loaded_metadata["fallback_reason"],
            "Temporary API failure",
        )
        self.assertEqual(loaded_metadata["records_extracted"], 1)
        self.assertTrue(loaded_metadata["warnings"])

    def test_corrupted_raw_json_raises_storage_error(self) -> None:
        """Corrupted JSON must not be used silently as cached data."""

        self.raw_directory.mkdir(parents=True, exist_ok=True)

        self.raw_file.write_text(
            "{invalid-json",
            encoding="utf-8",
        )

        self.metadata_file.write_text(
            json.dumps({"records_extracted": 1}),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            storage.StorageError,
            "invalid or corrupted",
        ):
            storage.load_cached_raw_snapshot("API unavailable")

    def test_cached_record_count_mismatch_raises_error(self) -> None:
        """Metadata and raw-data counts must agree before fallback use."""

        self.raw_directory.mkdir(parents=True, exist_ok=True)

        self.raw_file.write_text(
            json.dumps([{"recall_number": "D-1001"}]),
            encoding="utf-8",
        )

        self.metadata_file.write_text(
            json.dumps({"records_extracted": 2}),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            storage.StorageError,
            "count does not match",
        ):
            storage.load_cached_raw_snapshot("API unavailable")


if __name__ == "__main__":
    unittest.main()