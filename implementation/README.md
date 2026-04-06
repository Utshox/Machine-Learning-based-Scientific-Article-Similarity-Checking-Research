# Machine Learning-based Scientific Article Similarity Checking Research

This directory contains the thesis implementation for a hybrid plagiarism detector targeted at scientific articles. The current research workflow is centered on the official PAN 2025 generated plagiarism datasets.

## Research Objective
Detect semantic plagiarism, paraphrasing, structural reordering, and AI-assisted rewriting that lexical overlap methods often miss.

## Method
- Semantic layer: Sentence-BERT embeddings
- Structural layer: sentence and punctuation statistics
- Alignment layer: semantic Smith-Waterman dynamic programming
- Evaluation: character-level precision, recall, and F1

## Current Confirmed Result
The best confirmed Phase 2 configuration on 50 PAN validation pairs with `all-MiniLM-L6-v2` is stored in [`trained_config.json`](/Users/shinthiya.promi/Desktop/MS_THESIS/implementation/trained_config.json):

- precision: `0.4428`
- recall: `0.5466`
- F1: `0.4823`

## Data Layout
The code supports either zipped PAN archives or extracted directory layouts.

Expected raw archives under `data/`:
- `pan25-generated-plagiarism-detection-train.zip`
- `pan25-generated-plagiarism-detection-validation.zip`
- `pan25-generated-plagiarism-detection-spot-check.zip`

## Cloud-Friendly Runtime
The current scripts are now portable to Linux and GCP:

- `runtime_utils.py` resolves `cuda`, `mps`, or `cpu` automatically
- dataset paths can be passed explicitly with `--dataset`
- models can download online by default on first run
- `--offline` is available if model weights are already cached
- `test_mpnet.py` supports `--only-model` to run just MiniLM or just mpnet

## Recommended GCP Workflow
Use a Linux VM, install Python 3.10+, create a venv, install `requirements.txt`, and run one of:

```bash
python phase2_sweep.py 50 --workers 16
python test_mpnet.py 50 --device cuda --workers 1
python run_evaluation.py data/pan25-generated-plagiarism-detection-validation.zip 100 8 all-MiniLM-L6-v2
```

For the full GCP procedure, see [`GCP_RUNBOOK.md`](/Users/shinthiya.promi/Desktop/MS_THESIS/GCP_RUNBOOK.md).
