# Project Progress: Machine Learning-based Scientific Article Similarity Checking Research

This document tracks the progress, findings, and major changes in the development of the hybrid similarity checking model for the Master's Thesis.

## Project Overview
- **Objective:** Improve detection of semantic plagiarism (paraphrased, structural reordering, and AI-generated content) in scientific articles.
- **Approach:** A hybrid model combining **SBERT** (Sentence-BERT) for semantic embeddings and **Structural/Syntactic Features** for alignment.
- **Baseline:** Traditional methods like SVM and Lexical matching (N-gram, Levenshtein).

---

## Implementation Status

### Core Components
- [x] **Data Loader:** `data_loader.py` handles PAN-style datasets (txt + XML ground truth).
- [x] **Feature Extractor:** `features.py` extracts structural features (sentence count, avg length, POS tag ratios, etc.).
- [x] **Hybrid Scorer:** `scoring.py` implements a weighted sum of semantic similarity and structural difference.
- [x] **Evaluator:** `evaluator.py` implements sliding window detection and character-level metrics (Precision, Recall, F1).
- [x] **Mock Data Generator:** `mock_pan_generator.py` for local testing before obtaining official PAN datasets.

### Current Performance (Mock Data)
*Date: 2026-03-11*
- **Model:** `all-MiniLM-L6-v2` (SentenceTransformer)
- **Weights:** Semantic: 0.7, Structural: 0.3
- **Threshold:** 0.45 (Optimized)
- **Precision:** 0.4971
- **Recall:** 1.0000
- **F1 Score:** 0.6641
- **Finding:** Threshold tuning significantly improved Precision from 0.26 to 0.49. High Recall is maintained, but the model is still relatively sensitive. Further improvement likely requires more diverse mock data or the actual PAN dataset.

---

## Findings & Technical Notes
- **Semantic Sensitivity:** SBERT is very effective at catching paraphrased content even with high word-swapping.
- **Structural Alignment:** Adding POS tag ratios and sentence length helps differentiate between similar topics and actual structural copying.
- **Hardware:** Currently using **Apple Silicon (MPS)** for faster embedding generation.

---

## Major Changes
- **2026-04-06:**
    - Completed the 100-pair Phase 2 sweep on GCP after fixing multiple issues:
        - **Multiprocessing deadlock fix:** Patched `phase2_sweep.py` to bypass `mp.Pool` when `--workers 1`. The pool's `imap` was deadlocking near 93/100 pairs with both 16 and 4 workers.
        - **Numba JIT for Smith-Waterman:** Added `@njit` kernel in `evaluator.py`. The pure Python nested loop was taking 6+ hours for 72 combos on 100 pairs. Numba reduced it to ~49 minutes.
        - **OOM fix:** Merged Stage 1 (SW) and Stage 2 (post-processing) into a single pass with `del pair_raw_dets` after each combo. The old `sw_cache` dict stored all 72×100 raw detection lists (~32GB) and caused the OOM killer.
        - **GPU precompute:** Removed `CUDA_VISIBLE_DEVICES=''` and patched the workers=1 path to load the model once and reuse it. Precompute went from ~3 hours (CPU, per-pair model reload) to **15 minutes** (GPU, single model load).
    - 100-pair Phase 2 sweep results:
        - Best: sem=0.95, thr=0.70, gap=-0.5, chain=0.1, mindet=300
        - **F1=0.5258** (P=0.4898, R=0.5926)
        - Baseline F1=0.4712 → +11.6% improvement
    - Fresh venv created on VM (old one had hardcoded paths from `sinthiyanawsheen` user)
    - Model comparison completed (50 pairs, best Phase 2 config):
        - MiniLM: F1=0.4820, mpnet: F1=0.4620
        - **Winner: all-MiniLM-L6-v2** (faster and more accurate)
    - Uploaded 3 new experiment scripts to VM for further exploration:
        - `exp_fine_grid.py` — finer hyperparameter grid around the optimum
        - `exp_window_sweep.py` — window_size/step_size sweep (80–300 / 15–50)
        - `exp_train_val_split.py` — proper train/val methodology (tune on train, evaluate on val)
    - Current plan: run all 3 experiments sequentially on GCP GPU
- **2026-04-03:**
    - Stopped the local `all-mpnet-base-v2` validation comparison on macOS after confirming that the MPS + multiprocessing path was unreliable and unproductively slow for the larger model.
    - Preserved the current confirmed baseline from the 50-pair Phase 2 work:
        - `all-MiniLM-L6-v2`
        - Precision: **0.4428**
        - Recall: **0.5466**
        - F1: **0.4823**
    - Refactored the experiment scripts for cloud portability:
        - Added `implementation/runtime_utils.py` for device resolution (`cuda`/`mps`/`cpu`), dataset-path resolution, model loading, and safer CPU thread setup.
        - Updated `phase2_sweep.py`, `test_mpnet.py`, `run_evaluation.py`, `train_model.py`, and `optimize_hyperparams.py` to support cloud-style execution and normal model downloads.
        - Added `--dataset` and cloud-friendly runtime options where needed.
        - Added `--only-model` support to `test_mpnet.py` so MiniLM and mpnet can be run independently on GCP.
    - Added GCP handoff docs and helper scripts:
        - `GCP_RUNBOOK.md`
        - `implementation/gcp_setup.sh`
        - `implementation/gcp_phase2_examples.sh`
        - `implementation/unpack_pan_data.sh`
    - Decision: move the remaining Phase 2 work to GCP, where larger Linux CPU/GPU machines can be used to rerun the sweep, confirm mpnet properly, and expand final validation beyond the local machine limits.
    - GCP VM setup completed enough to run experiments:
        - VM user: `sinthiyanawsheen`
        - GPU confirmed working via `nvidia-smi`: **NVIDIA L4**
        - Python 3.11, `tmux`, and `unzip` installed
        - Bundle uploaded and unpacked on the VM:
            - `~/ms_thesis_gcp_bundle.zip`
            - `~/MS_THESIS_GCP_READY/implementation`
    - Important GCP runtime findings:
        - Browser SSH works reliably; direct custom-key SSH from the local terminal was intermittent and should not be relied on for handoff.
        - `phase2_sweep.py` with many workers on a single GPU is a bad configuration because every worker auto-selects the same `cuda:0`.
        - NLTK resources had to be installed manually on the VM:
            - `stopwords`
            - `punkt`
            - `punkt_tab`
            - `averaged_perceptron_tagger`
            - `averaged_perceptron_tagger_eng`
    - Current in-progress GCP run at handoff:
        - Running from browser SSH inside `tmux` session `thesis`
        - Command family used:
            - `CUDA_VISIBLE_DEVICES='' python phase2_sweep.py 100 --workers 16 --offline`
        - This run advanced to **93/100** in the precompute stage, but looked potentially tail-heavy / partially stuck near the end.
        - Recommended decision rule for the next agent:
            - wait 10-15 minutes if progress is still moving
            - if it remains frozen at the same count, stop it and rerun with fewer workers:
              `CUDA_VISIBLE_DEVICES='' python phase2_sweep.py 100 --workers 4 --offline`
    - Immediate next plan on GCP:
        - finish the 100-pair CPU Phase 2 sweep
        - inspect `phase2_results.json` and `trained_config.json`
        - run `python test_mpnet.py 50 --device cuda --workers 1 --offline`
        - run final validation on 100+ pairs with the winning model
- **2026-04-02:**
    - Generated 6 thesis-quality visualization figures in `implementation/figures/` for Chapter 4.
    - Ran hyperparameter sweep on 10 PAN 2025 validation pairs across 20 configurations (5 semantic weights × 4 thresholds).
    - Key finding: optimal threshold on validation data is **0.65** (not 0.55), with best F1=0.506 at sem=0.90/thr=0.65.
    - Precision-Recall curve confirms threshold 0.55 is suboptimal — F1 peaks at 0.65–0.70.
    - Structural contribution analysis confirms hybrid model outperforms semantic-only at most thresholds, validating the thesis methodology.
    - Detection overlay visualization clearly shows the false positive problem (model over-triggers on clean regions).
    - Wrote complete thesis in LaTeX (`thesis_part2/main.tex`) — 5 chapters, 6 figures, 7 equations, 4 tables, 29 APA references. Includes Part 1 content integrated.
    - **Performance optimisation of codebase:**
        - Vectorised Smith-Waterman DP in `evaluator.py` — **5.4x speedup** (row-level NumPy + bounded left-gap scan).
        - Vectorised structural similarity matrix using NumPy broadcasting (eliminated O(m*n) Python loop).
        - Added `multiprocessing.Pool` to `run_evaluation.py` and `train_model.py` for parallel pair processing across all CPU cores.
        - `evaluator.py` now supports tuneable `chain_threshold` and `min_detection_length` parameters.
    - **Phase 2 sweep launched:** `phase2_sweep.py` running on 50 PAN validation pairs with 2,160 configs (4 sem_weights × 6 thresholds × 3 gap_penalties × 6 chain_thresholds × 5 min_det_lengths). Two-stage approach: expensive SW runs once per (sem_w, thr, gap) combo, then post-processing params swept for free.
    - **SBERT upgrade:** `all-mpnet-base-v2` model downloaded and cached, pending testing after sweep completes.
    - Status at session end: sweep running in background (~50 min remaining).
- **2026-04-01:**
    - Added dedicated thesis-level experiment logging in `EXPERIMENT_LOG.md`.
    - Added automatic summary logging for completed training and evaluation runs via `implementation/experiment_logger.py`.
    - Stopped the impractical full 62,160-pair streaming train sweep after confirming it would take on the order of months with the current Python-heavy Smith-Waterman pipeline.
    - Added on-disk pair-label caching (`.pair_labels_01_train.json`) so balanced subset selection on the PAN train split no longer needs to rescan XML truth files on every run.
    - Completed a balanced staged CPU training run on **500 official PAN 2025 train pairs**:
        - Best semantic weight: **0.70**
        - Best structural weight: **0.30**
        - Best threshold: **0.55**
        - Macro F1: **0.3180**
    - Completed a balanced staged CPU training run on **2,000 official PAN 2025 train pairs**:
        - Best semantic weight: **0.80**
        - Best structural weight: **0.20**
        - Best threshold: **0.55**
        - Precision: **0.2825**
        - Recall: **0.4720**
        - Macro F1: **0.3393**
    - Current best saved training configuration in `implementation/trained_config.json`:
        - `semantic_weight = 0.80`
        - `structural_weight = 0.20`
        - `threshold = 0.55`
        - `window_size = 150`
        - `step_size = 25`
    - Began bounded hyperparameter optimization beyond weights/threshold by adding `optimize_hyperparams.py`, which searches:
        - `window_size`
        - `step_size`
        - `semantic_weight`
        - `threshold`
    - Current optimizer status at session handoff:
        - Running on a balanced **500-pair** PAN train subset
        - First layout under evaluation is `window_size=120`, `step_size=20`
        - This branch is computationally heavy and may be worth narrowing to more practical layouts such as `150/25`, `180/25`, and `180/30`
- **2026-03-31:**
    - Added support for loading the official **PAN 2025** train, validation, and spot-check datasets directly from the provided `.zip` archives. No manual extraction or reshaping into the old mock `pairs/src/susp` format is now required.
    - Added support for loading the official extracted PAN directory layout as well, including sibling `*_truth` folders.
    - Added a dedicated `train_model.py` step to fit the hybrid scorer parameters on the official **PAN 2025 train split** and save the learned configuration into `trained_config.json`.
    - Refactored training to precompute pair-level semantic and structural matrices once, then sweep thresholds and weights over the cached pair data. This made CPU training practical and enabled live `tqdm` progress for label scanning, pair precomputation, and configuration search.
    - Updated `run_evaluation.py` and `tuning.py` to target the real PAN archives by default and accept optional CLI arguments for dataset path and pair limits.
    - Updated evaluation scripts to read trained parameters from disk instead of always using the original hardcoded mock-data defaults.
    - Switched SentenceTransformer loading to offline/local-cache mode so evaluation can run without network access when the cached model is already present.
    - Completed a balanced CPU training run on **6 official PAN 2025 train pairs** (3 positive, 3 negative) using the extracted train split:
        - Best semantic weight: **0.80**
        - Best structural weight: **0.20**
        - Best threshold: **0.55**
        - Macro F1 on the bounded train subset: **0.3310**
    - Completed a validation smoke test on **1 official PAN 2025 validation pair**:
        - Precision: **0.4313**
        - Recall: **1.0000**
        - F1 Score: **0.6027**
- **2026-03-11:**
    - **SOTA Alignment:** Implemented **Semantic Smith-Waterman** alignment in `evaluator.py`. The model now looks for the "best path" through the document using dynamic programming, handling scrambled sequences.
    - **Performance Optimization:** Tuned thresholds to 0.45, improving Precision to 0.49 on mock data.
    - Established initial evaluation pipeline.
    - Created `PROGRESS.md` for session continuity.

## Session Continuity & Instructions
**IMPORTANT FOR NEXT AGENT:**
- **Methodology Lock:** Do NOT change the core architecture (SBERT + Structural Features + Smith-Waterman Alignment). This methodology was approved in **Thesis Part 1** and must be maintained. Upgrading SBERT model variant (MiniLM → mpnet) is allowed — same architecture, better weights.
- **Context Efficiency:** Read this `PROGRESS.md` first. Then inspect the current GCP run state rather than assuming local-only work. The evaluator is already optimised — do not rewrite it.
- **Active environment:** The current active environment is the **GCP VM**, not the local Mac. SSH via `gcloud compute ssh instance-20260403-000831 --zone=us-central1-a`.
- **Active directory on VM:** `~/MS_THESIS_GCP_READY/implementation`
- **VM user:** `shinthiya.promi` (changed from `sinthiyanawsheen`; files copied to new user home)
- **Phase 2 sweep:** COMPLETED. Results in `phase2_results.json` and `trained_config.json`.
- **Model comparison:** COMPLETED. MiniLM confirmed winner. Results in `model_comparison.json`.
- **Remaining:** Final validation on 100+ pairs, re-generate visualizations, update thesis LaTeX.
- **Thesis LaTeX:** Located at `thesis_part2/main.tex` with `references.bib` and `images/`. Uses `natbib`+`apalike` (NOT biblatex). Architecture diagram is TikZ inline. Update Chapter 4 numbers after Phase 2.
- **Code state:** `evaluator.py` now supports `chain_threshold` and `min_detection_length` parameters. `run_evaluation.py` and `train_model.py` support multiprocessing. The local Mac venv is at `implementation/venv/` (Python 3.9), but the GCP VM uses its own venv under `~/MS_THESIS_GCP_READY/implementation/venv` (Python 3.11).
- **Cache Awareness:** Reuse `implementation/.pair_labels_01_train.json` for future staged train runs.
- **Logging Awareness:** Keep `EXPERIMENT_LOG.md` updated. `phase2_sweep.py` auto-logs to it.
- **Goal:** Finalise Phase 2 results on GCP, confirm best config on 100+ pairs, update thesis, generate final visualizations.

---

## Next Steps (Phase 2 — resume here)
1. [x] **Phase 1 Visualizations:** 6 thesis-quality figures generated in `implementation/figures/`.
2. [x] **Thesis LaTeX:** Complete 5-chapter document in `thesis_part2/main.tex`.
3. [x] **Code optimisation:** Vectorised evaluator (5.4x SW speedup), multiprocessing added.
4. [x] **Finish current GCP Phase 2 sweep:** Completed 100-pair sweep on GCP. Best F1=0.5258 (sem=0.95, thr=0.70, gap=-0.5, chain=0.1, mindet=300). Fixed multiprocessing deadlock, added Numba JIT (~100x SW speedup), GPU precompute (~13x faster).
5. [x] **Test mpnet model on GCP:** MiniLM wins (F1=0.4820 vs mpnet F1=0.4620 on 50 pairs with best config).
6. [x] **Experiment 5 — Fine grid:** F1=0.5263 (+0.0005), confirms optimum is converged at sem~0.94-0.98, thr~0.70.
7. [x] **Experiment 1 — Window sweep:** Window 200/25 wins with F1=0.5291 (+0.0033 over 150/25). Step size 25 consistently best. 2h08m runtime.
8. [x] **Experiment 3 — Train/val split:** Train-tuned F1=0.5219 on val (vs val-tuned 0.5258). Only -0.0038 drop — model generalizes well, not overfit.
9. [ ] **Final validation:** Run the overall best config on 100+ validation pairs (or spot-check set if train/val split is used).
10. [ ] **Re-generate visualizations:** Update figures with improved config for Chapter 4.
11. [ ] **Update thesis LaTeX:** Add all experiment results, update tables, conclusions.
