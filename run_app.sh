#!/usr/bin/env bash
set -euo pipefail

conda run -n project1 python -m streamlit run app.py "$@"