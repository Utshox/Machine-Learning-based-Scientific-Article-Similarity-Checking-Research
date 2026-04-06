#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
Example commands for GCP:

1. Phase 2 sweep on 100 validation pairs
python phase2_sweep.py 100 --workers 16

2. MiniLM vs mpnet comparison on GPU
python test_mpnet.py 50 --device cuda --workers 1

3. mpnet-only comparison on GPU
python test_mpnet.py 50 --device cuda --workers 1 --only-model all-mpnet-base-v2

4. Final validation on 100 pairs
python run_evaluation.py data/pan25-generated-plagiarism-detection-validation.zip 100 8 all-MiniLM-L6-v2
EOF
