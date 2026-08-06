"""Central configuration for the Drug Recall Intelligence Dashboard."""

from datetime import date
from pathlib import Path


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# openFDA API settings
API_BASE_URL = "https://api.fda.gov/drug/enforcement.json"

START_DATE = "20210101"
END_DATE = date.today().strftime("%Y%m%d")

PAGE_SIZE = 1000
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
MAX_API_SKIP = 25_000


# Analytics settings
ROLLING_WINDOW_MONTHS = 3

EXPECTED_CLASSIFICATIONS = {
    "Class I",
    "Class II",
    "Class III",
    "Not Yet Classified",
}


# Expected fields from the API
EXPECTED_API_FIELDS = {
    "recall_number",
    "event_id",
    "classification",
    "recalling_firm",
    "product_description",
    "reason_for_recall",
    "voluntary_mandated",
    "distribution_pattern",
    "city",
    "state",
    "country",
    "recall_initiation_date",
    "report_date",
    "termination_date",
    "status",
}


# Output file paths
RAW_DATA_FILE = RAW_DATA_DIR / "drug_recalls_raw.json"
EXTRACTION_METADATA_FILE = RAW_DATA_DIR / "extraction_metadata.json"

RECALLS_OUTPUT_FILE = PROCESSED_DATA_DIR / "recalls_enriched.csv"
MONTHLY_SUMMARY_FILE = PROCESSED_DATA_DIR / "monthly_summary.csv"
FIRM_SUMMARY_FILE = PROCESSED_DATA_DIR / "firm_summary.csv"
CLASSIFICATION_DIM_FILE = PROCESSED_DATA_DIR / "dim_classification.csv"
DATA_QUALITY_REPORT_FILE = PROCESSED_DATA_DIR / "data_quality_report.json"