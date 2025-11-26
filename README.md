# FME-UPC Datathon 2025

Smadex-style two-stage LTV prediction stack with Dask-based data loading, histogram samplers, LightGBM models, and a Streamlit monitoring UI.

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) for dependency and environment management

Install uv (one time) if you do not already have it available:

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

## Setup

Synchronize the project environment (creates a `.venv` managed by uv and installs everything from `pyproject.toml`):

```bash
uv sync
```

## Common workflows

Train models and persist artifacts under `models/`:

```bash
uv run python scripts/train.py --config config/config.yaml
```

Generate a submission file (expects parquet inputs defined in the config):

```bash
uv run python scripts/predict.py --config config/config.yaml --dataset data/raw/test --output data/submissions/submission.csv
```

Launch the monitoring dashboard:

```bash
uv run streamlit run frontend/app.py
```

See `IMPLEMENTATION_GUIDE.md` for the full architecture, data flow, and evaluation strategy.
