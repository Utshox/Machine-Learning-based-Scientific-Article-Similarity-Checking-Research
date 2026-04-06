#!/usr/bin/env bash
set -euo pipefail

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p "${HOME}/.cache/sentence-transformers"
echo "GCP environment ready."
