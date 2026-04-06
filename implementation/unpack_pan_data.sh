#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/data"

mkdir -p extracted

if [ -f pan25-generated-plagiarism-detection-train.zip ]; then
  unzip -q -o pan25-generated-plagiarism-detection-train.zip -d extracted/01_train
fi

if [ -f pan25-generated-plagiarism-detection-validation.zip ]; then
  unzip -q -o pan25-generated-plagiarism-detection-validation.zip -d extracted/02_validation
fi

if [ -f pan25-generated-plagiarism-detection-spot-check.zip ]; then
  unzip -q -o pan25-generated-plagiarism-detection-spot-check.zip -d extracted/03_spot_check
fi

echo "PAN archives unpacked under implementation/data/extracted."
