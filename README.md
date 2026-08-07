# Drug Recall Intelligence

Drug Recall Intelligence is a historical analytics product built with Python,
pandas, Streamlit and Plotly. It extracts official drug enforcement reports
from the [openFDA Drug Enforcement API](https://open.fda.gov/apis/drug/enforcement/),
builds auditable local data products, and presents reporting patterns through
an interactive dashboard.

This project is not medical advice, a live recall-alert service, or a source of
current recall status.

## Live Demo

[Open the deployed Streamlit dashboard](https://dario-drug-recall-intelligence.streamlit.app/)

The GitHub repository remains the source of truth and the submission artifact.
The hosted app is a convenient way to review the committed dashboard and data
snapshot.

## What the Product Does

The dashboard helps a business or data reviewer explore:

- historical enforcement-report volume and trends;
- FDA classification mix;
- time between recall initiation and report publication;
- firm-level record concentration, without treating volume as a safety score;
- searchable report-level product and recall details; and
- extraction provenance, reconciliation and data-quality findings.

## Architecture

```text
Official openFDA Drug Enforcement API
                  |
                  v
        paginated extraction
                  |
                  v
     raw JSON snapshot + metadata
                  |
                  v
 transformation and analytical modeling
                  |
                  v
       data-quality validation
                  |
                  v
 monthly + firm analytical summaries
                  |
                  v
      committed processed outputs
                  |
                  v
    Streamlit + Plotly dashboard
```

| Component | Responsibility |
| --- | --- |
| `src/api_client.py` | Validates the date query, paginates the openFDA endpoint, applies bounded retries, and records extraction metadata. |
| `src/storage.py` | Persists raw JSON and processed CSV/JSON files using temporary files and atomic per-file replacement; validates cached raw snapshots. |
| `src/transform.py` | Cleans source fields, parses dates, normalizes firm names conservatively, creates deterministic identifiers, flags duplicates/anomalies, derives date metrics, and builds the classification dimension. |
| `src/quality.py` | Produces structured schema, completeness, duplicate, date, classification and row-count checks with error, warning and information severities. |
| `src/analytics.py` | Builds a de-duplicated analytical base, continuous monthly summaries, rolling averages and normalized-firm summaries. |
| `etl.py` | Orchestrates extraction, transformation, validation, analytics and processed-output persistence. |
| `app.py` | Validates generated assets and renders the filtered Streamlit analytics product. |

## Data Model and Outputs

The ETL writes the following local artifacts:

| File | Purpose |
| --- | --- |
| `data/raw/drug_recalls_raw.json` | Original openFDA result objects retained as the auditable raw snapshot. |
| `data/raw/extraction_metadata.json` | Query, pagination, counts, timestamps, warnings and live/cache extraction status. |
| `data/processed/recalls_enriched.csv` | Enriched report-level fact table with source identifiers, cleaned text, normalized firm, parsed dates, derived durations and quality flags. |
| `data/processed/monthly_summary.csv` | Continuous monthly record counts, classification counts, firm counts, reporting lag, termination-date coverage and three-month rolling average. |
| `data/processed/firm_summary.csv` | Record, event, Class I, date-range, reporting-lag and active-year metrics by normalized firm. |
| `data/processed/dim_classification.csv` | Classification order, severity label and established description used by both transformation and dashboard context. |
| `data/processed/data_quality_report.json` | Structured validation findings, reconciliation, quality status and pipeline context. |

Important identifiers have different meanings:

- An **enforcement report record** is one result row returned by the API. It is
  the dashboard's basic record-count unit and is not necessarily a unique
  incident.
- `recall_number` is the source recall identifier. It can be missing or appear
  on more than one report row, so it is not used as a guaranteed row key.
- `event_id` is the source event identifier. The unique recall-event metric is
  the count of distinct non-missing `event_id` values.
- `recalling_firm_normalized` standardizes case, punctuation and whitespace.
  It deliberately avoids fuzzy matching and legal-suffix removal so that
  distinct organizations are not merged aggressively.

Exact source duplicates remain in `recalls_enriched.csv` for auditability and
are flagged. Business metrics use one row per deterministic
`source_record_hash`, so exact duplicates do not inflate analytical totals.

## Key Metrics

The committed snapshot was extracted on 2026-08-07 for reports from 2021-01-01
through 2026-08-07.

| Metric | Current snapshot |
| --- | ---: |
| Enforcement report records | 5,527 |
| Unique recall events | 1,752 |
| Recalling firms | 657 |
| Class I records | 463 |
| Median reporting lag | 33 days |
| Termination-date coverage | 57.2% |

openFDA is a changing public source. These metrics may change when
`python etl.py` refreshes the snapshot.

## Important Data Semantics

- `status_at_publication` is the status contained in the enforcement report
  when it was published. It is historical context, not current recall status.
- openFDA explicitly says enforcement reports should not be used for public
  alerts or recall lifecycle tracking, and that published enforcement status
  is not subsequently maintained as a live status. See the
  [official Drug Enforcement API disclaimer](https://open.fda.gov/apis/drug/enforcement/#disclaimer).
- Missing `termination_date` values are contextual. They are reported as an
  informational coverage finding, not automatically as a quality failure.
- `reporting_lag_days` is `report_date - recall_initiation_date`. Negative
  results are flagged and excluded from the metric.
- `termination_days` is `termination_date - recall_initiation_date` and is
  calculated only when the dates are present and logically valid.
- A latest month that ends before calendar month-end is explicitly marked
  partial in the dashboard and is excluded from the displayed three-month
  rolling-average endpoint.
- The data is historical public enforcement-report data only. It must not be
  used as medical guidance or as a firm safety/risk ranking.

## Data Quality and Reliability

The quality layer checks:

- required transformed columns and critical-field completeness;
- unique `record_id` values and exact source duplicates;
- repeated non-missing recall numbers without deleting them;
- invalid initiation, report and termination dates;
- negative reporting-lag and termination-duration calculations;
- expected classifications and classification-dimension joins;
- extracted versus transformed row counts; and
- API-reported totals versus extracted counts.

Structural errors produce a failed quality status and stop analytical output
creation. Non-fatal source-quality or reconciliation issues produce warnings.
Contextual observations, such as termination-date coverage, are informational.
An API total mismatch is a warning rather than an automatic failure because the
source result set can change during pagination.

The committed quality report is **Passed with warnings**:

| Finding | Result |
| --- | ---: |
| Errors | 0 |
| Warnings | 1 |
| Informational findings | 1 |
| Records missing `recall_number` | 2 |
| Records with contextual missing `termination_date` | 2,368 |
| API reported / extracted / transformed records | 5,527 / 5,527 / 5,527 |

### Reliability and API Behavior

- Requests are sorted and paginated in pages of 1,000 within openFDA's
  supported skip range.
- Network failures, HTTP 429 and retryable server errors use up to three
  attempts with bounded backoff; a numeric `Retry-After` header is respected.
- An openFDA HTTP 404 with error code `NOT_FOUND` represents a valid empty
  result set. Other HTTP errors fail clearly.
- An optional `OPENFDA_API_KEY` environment variable is supported, but no API
  key is required.
- Expected live-extraction failures (`OpenFDAClientError`) may use a validated
  local raw snapshot. Unexpected programming or configuration errors propagate
  instead of being silently converted into cached fallback.
- Transformation, quality validation and analytical summaries are calculated
  in memory before processed files begin replacing successful outputs.
- Each JSON or CSV write uses a temporary file followed by atomic replacement.

This is still a local file pipeline, not a transactional database. The
processed output set is written sequentially, so it does not provide a single
multi-file commit boundary. A successful live extraction is saved before
downstream processing; therefore the raw snapshot can be newer than processed
outputs if transformation, quality, analytics or later output writing fails.

## Dashboard

The Streamlit application has four areas:

- **Overview** — primary KPIs, operational context, interactive monthly trend,
  classification mix, filtered insights and classification definitions.
- **Firm Intelligence** — volume ranking with record/event context and optional
  selected-firm history. Volume is explicitly not presented as risk.
- **Recall Explorer** — searchable historical report records, configurable row
  count and CSV download.
- **Data Quality** — business-readable findings, reconciliation and extraction
  provenance, with technical JSON details available separately.

Global filters cover report date, classification, status at publication and
normalized firm search. Plotly provides hover, legend, zoom, pan, reset and
download interactions. The app validates required files and essential columns
before rendering and displays an actionable error if assets are missing or
incompatible.

## Setup

Python 3.10 or 3.11 is recommended.

1. Clone and enter the repository:

   ```bash
   git clone https://github.com/AVR314/Dario-Ai-data-engineer-assignment.git
   cd Dario-Ai-data-engineer-assignment
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate it.

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Run the ETL:

   ```bash
   python etl.py
   ```

6. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

No openFDA API key is required. If available, `OPENFDA_API_KEY` can be set in
the environment and will be added to API requests automatically.

## Running Without a Fresh ETL

Committed processed snapshots make the repository directly reviewable:

```bash
streamlit run app.py
```

Running `python etl.py` first is recommended when network access is available
and a fresh snapshot is desired. Starting the dashboard from committed
processed files is separate from the ETL's raw-snapshot fallback behavior.

## Testing

Run the complete suite with:

```bash
python -m unittest discover -s tests -v
```

Current verified result: **71 tests passing**.

The suite covers API pagination and failure behavior, raw-snapshot storage,
transformation, data quality, analytical summaries, ETL orchestration, and pure
dashboard logic including filtering, duplicate handling and Plotly figure
configuration. It does not claim browser-automation coverage.

## AI-Assisted Development

AI was used as a development partner for requirement decomposition,
architecture discussion, implementation assistance, debugging, assumption
review, openFDA semantic validation, test strategy and UI refinement. The work
also included challenging and correcting earlier AI-generated decisions rather
than accepting them uncritically.

## Assumptions and Limitations

- The product uses public historical drug enforcement reports only.
- The source can add, revise or remove records between ETL runs.
- It does not provide medical guidance, public alerts or current-status
  tracking.
- Firm normalization is intentionally conservative; there is no fuzzy entity
  resolution or corporate-family matching.
- There are no external joins to product, company or other FDA datasets.
- Individual file replacement is atomic, but the local CSV/JSON output set is
  not a database transaction.
- A newly saved raw snapshot can be newer than the processed outputs if a
  downstream stage fails.
- Firm and classification metrics describe representation in this report
  dataset, not causal performance, medical severity beyond the source
  classification, or company safety.

## Future Improvements

- Add carefully selected FDA enrichment sources while preserving source-level
  provenance.
- Introduce stronger, reviewable organization/entity resolution.
- Schedule and monitor refreshes through an orchestration service.
- Move analytical outputs to a durable warehouse with transactional publishing.
- Add richer period-over-period historical comparisons.

## Repository Structure

```text
.
├── .gitignore
├── .streamlit/
│   └── config.toml
├── README.md
├── app.py
├── etl.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── analytics.py
│   ├── api_client.py
│   ├── config.py
│   ├── quality.py
│   ├── storage.py
│   └── transform.py
├── tests/
│   ├── test_analytics.py
│   ├── test_api_client.py
│   ├── test_app.py
│   ├── test_etl.py
│   ├── test_quality.py
│   ├── test_storage.py
│   └── test_transform.py
├── data/
│   ├── raw/
│   │   ├── .gitkeep
│   │   ├── drug_recalls_raw.json
│   │   └── extraction_metadata.json
│   └── processed/
│       ├── .gitkeep
│       ├── data_quality_report.json
│       ├── dim_classification.csv
│       ├── firm_summary.csv
│       ├── monthly_summary.csv
│       └── recalls_enriched.csv
```
