# ORACL — Research Proposal Dataset Pipeline

ORACL scrapes published research papers, converts them into structured research proposals via LLM, and optionally grades them via pairwise comparison. It supports two data sources:

- **arXiv** — works for all conferences (CVPR, ICCV, ECCV, NeurIPS, ACL, etc.)
- **OpenReview** — works for conferences hosted on the platform (ICLR, NeurIPS, ICML, ACL, EMNLP, NAACL, AAAI) and includes **accept/reject labels**

## Setup

### 1. Create/update the conda environment

```bash
conda env create -f ai_researcher/environment.yml
conda activate ai-researcher
```

Or update an existing environment:

```bash
conda env update -f ai_researcher/environment.yml --prune
```

### 2. Set your API key

Create (or edit) `ai_researcher/config/.env`:

```
OPENAI_API_KEY=sk-...
```

The pipeline uses `gpt-5-mini` for proposal conversion and `gpt-5.2` for grading.

## Usage

All commands are run from the **repo root** (`idea-generation-agent/`).

### Scrape + Convert (arXiv)

Fetches papers from arXiv by category and date, downloads PDFs, extracts text, and converts to proposals via LLM.

```bash
# Basic usage
python -m ORACL.main --conference CVPR --year 2023

# With options
python -m ORACL.main --conference NeurIPS --year 2024 --month_start 1 --month_end 6 --max_papers 50

# Fast mode (abstracts only, no PDF download)
python -m ORACL.main --conference ACL --year 2024 --skip_pdf
```

### Scrape + Convert (OpenReview)

Fetches papers directly from OpenReview with **accept/reject labels**. Supports balanced sampling to ensure data diversity.

```bash
# Basic usage
python -m ORACL.main --openreview --conference ICLR --year 2024

# Balanced 50/50 accept/reject, max 100 papers
python -m ORACL.main --openreview --conference NeurIPS --year 2023 --max_papers 100 --accepted_ratio 0.5

# Only papers from June onward, skip PDFs
python -m ORACL.main --openreview --conference ICML --year 2024 --min_month 6 --skip_pdf
```

### Grade Proposals

Runs pairwise comparison grading on an existing dataset using GPT-5.2 with web search.

```bash
python -m ORACL.main --grade --conference CVPR --year 2023

# Limit number of pairs (useful for large datasets)
python -m ORACL.main --grade --conference ICLR --year 2024 --max_pairs 20
```

### List Datasets

```bash
python -m ORACL.main --list
```

## CLI Reference

| Flag | Description | Default |
|---|---|---|
| `--conference` | Conference name (see tables below) | `ACL` |
| `--year` | Publication year | `2024` |
| `--max_papers` | Max papers to fetch | `5` |
| `--skip_pdf` | Use abstracts only (faster) | off |
| `--month_start` | Start month (arXiv pipeline) | `1` |
| `--month_end` | End month (arXiv pipeline) | `12` |
| `--search_query` | Extra arXiv search terms | none |
| `--openreview` | Use OpenReview instead of arXiv | off |
| `--min_month` | Min month filter (OpenReview) | `1` |
| `--accepted_ratio` | Target fraction of accepted papers (OpenReview) | `0.5` |
| `--grade` | Grade an existing dataset | off |
| `--max_pairs` | Max pairs for grading | all |
| `--grading_model` | Model for grading | `gpt-5.2` |
| `--list` | List all stored datasets | off |

## Supported Conferences

### arXiv pipeline (all conferences)

| Conference | arXiv Category |
|---|---|
| ACL, EMNLP, NAACL | `cs.CL` |
| NeurIPS, ICML, ICLR | `cs.LG` |
| AAAI | `cs.AI` |
| CVPR, ICCV, ECCV | `cs.CV` |

### OpenReview pipeline (with accept/reject labels)

| Conference | Supported Years |
|---|---|
| ICLR | 2021–2024 |
| NeurIPS | 2021–2024 |
| ICML | 2023–2024 |
| AAAI | 2025 |
| ACL | 2024 |
| EMNLP | 2024 |
| NAACL | 2024 |

> **Note:** CVPR, ICCV, and ECCV use CMT (not OpenReview), so accept/reject labels are not available for those. Use the arXiv pipeline instead.

## Output Format

Proposals are stored in `ORACL/dataset/<CONFERENCE>/<YEAR>.jsonl`, one JSON object per line:

```json
{
  "title": "...",
  "problem": "...",
  "motivation": "...",
  "proposed_method": "...",
  "experiment_plan": "...",
  "source_arxiv_id": "2401.12345",
  "source_openreview_id": "",
  "accepted": true
}
```

- `source_arxiv_id` — populated for arXiv-sourced papers
- `source_openreview_id` — populated for OpenReview-sourced papers
- `accepted` — `true`/`false` for OpenReview papers, `null` for arXiv papers

Grading results are saved to `ORACL/grading_results/<CONFERENCE>/<YEAR>_<TIMESTAMP>/`.

## Project Structure

```
ORACL/
├── main.py                  # CLI entry point & pipeline orchestration
├── config/
│   └── settings.py          # All configuration (models, paths, defaults)
├── modules/
│   ├── arxiv_scraper.py     # arXiv paper fetching & PDF extraction
│   ├── openreview_scraper.py # OpenReview fetching with accept/reject labels
│   ├── converter.py         # LLM-based paper → proposal conversion
│   ├── dataset_store.py     # JSONL dataset read/write/dedup
│   └── grader.py            # Pairwise proposal grading (GPT-5.2 + web search)
├── dataset/                 # Generated JSONL datasets
└── grading_results/         # Grading outputs
```
