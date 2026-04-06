# Experiment Log

This file records bounded training, tuning, and validation results in chronological order for later thesis writing.

## Initial Notes
- Scope: PAN 2025 train/validation experiments for the approved hybrid SBERT + structural + Smith-Waterman method.
- Current best saved config before automatic logging integration:
  - semantic_weight: 0.80
  - structural_weight: 0.20
  - threshold: 0.55
  - window_size: 150
  - step_size: 25
  - F1: 0.3393 on the 2,000-pair balanced train subset

## Validation Snapshot
- Date: 2026-04-01
- Dataset: PAN 2025 validation
- Pair limit: 100
- Status: In progress during session
- Early observation:
  - Pair 1 (positive): Precision 0.5107, Recall 0.9730, F1 0.6698
  - Pair 2 (negative): 46 false detections, Precision 0.0000, Recall 0.0000, F1 0.0000
- Interpretation: current config is strong on positive recall but still over-triggers on clean documents.

## Visualization & Validation Run (10-pair validation subset)
- Date: 2026-04-02
- Dataset: PAN 2025 validation (10 pairs)
- Config: semantic_weight=0.80, structural_weight=0.20, threshold=0.55, window_size=150, step_size=25
- Hyperparameter sweep results (Fig 3):
  - Best F1 on validation subset: **0.506** at semantic_weight=0.90, threshold=0.65
  - Current config (sem=0.80, thr=0.55) achieves F1=0.467 on same subset
  - Threshold 0.65 consistently outperforms 0.55 across all semantic weights
- Precision-Recall curve findings (Fig 4):
  - Precision rises steeply: 0.30 at thr=0.20 → 0.51 at thr=0.75
  - Recall stable at ~0.70 until thr=0.55, then drops sharply
  - F1 peaks around threshold 0.65–0.70
- Structural contribution findings (Fig 6):
  - Hybrid model (sem=0.80, str=0.20) outperforms semantic-only at thresholds 0.30–0.60
  - At threshold 0.70, semantic-only slightly edges ahead
  - Confirms structural features add value at moderate thresholds
- 6 thesis-quality figures generated in `implementation/figures/`:
  1. `fig1_semantic_heatmap.png` — SBERT cosine similarity matrix
  2. `fig2_alignment_path.png` — Smith-Waterman DP score matrix with traceback
  3. `fig3_hyperparam_sweep.png` — F1 across semantic_weight × threshold grid
  4. `fig4_pr_curve.png` — Precision/Recall/F1 vs threshold curve
  5. `fig5_detection_overlay.png` — Character-level ground truth vs prediction overlay
  6. `fig6_structural_contribution.png` — Hybrid vs semantic-only baseline comparison

## Validation Run
- Date: 2026-04-02 04:50:22
- dataset: /Users/shinthiya.promi/Desktop/MS_THESIS/implementation/data/pan25-generated-plagiarism-detection-validation.zip
- pair_limit: 6
- workers: 6
- semantic_weight: 0.8000
- structural_weight: 0.2000
- threshold: 0.5500
- window_size: 150
- step_size: 25
- avg_precision: 0.2471
- avg_recall: 0.4852
- avg_f1: 0.3273
## Validation Run
- Date: 2026-04-02 04:51:39
- dataset: /Users/shinthiya.promi/Desktop/MS_THESIS/implementation/data/pan25-generated-plagiarism-detection-validation.zip
- pair_limit: 6
- workers: 1
- semantic_weight: 0.8000
- structural_weight: 0.2000
- threshold: 0.5500
- window_size: 150
- step_size: 25
- avg_precision: 0.2471
- avg_recall: 0.4852
- avg_f1: 0.3273
## Phase 2 Sweep
- Date: 2026-04-02 19:51:46
- dataset: /Users/shinthiya.promi/Desktop/MS_THESIS/implementation/data/pan25-generated-plagiarism-detection-validation.zip
- pair_limit: 50
- configs_tested: 2160
- baseline_f1: 0.4244
- best_semantic_weight: 0.9500
- best_threshold: 0.7000
- best_chain_threshold: 0.1000
- best_gap_penalty: -0.5000
- best_min_detection_length: 300
- best_precision: 0.4428
- best_recall: 0.5466
- best_f1: 0.4823
## Model Comparison
- Date: 2026-04-02 21:43:12
- dataset: /Users/shinthiya.promi/Desktop/MS_THESIS/implementation/data/pan25-generated-plagiarism-detection-validation.zip
- pair_limit: 1
- config: {'semantic_weight': 0.95, 'structural_weight': 0.05, 'threshold': 0.7, 'chain_threshold': 0.1, 'gap_penalty': -0.5, 'min_detection_length': 300, 'window_size': 150, 'step_size': 25, 'precision': 0.44279953372359115, 'recall': 0.5466290699361012, 'f1': 0.48225469969933066}
- minilm_f1: 0.7623
- mpnet_f1: 0.7498
- winner: all-MiniLM-L6-v2

## GCP Migration
- Date: 2026-04-03
- Status: Switched remaining Phase 2 work from the local Mac to a GCP GPU VM.
- Bundle created locally:
  - `/Users/shinthiya.promi/Desktop/MS_THESIS/GCP_READY/MS_THESIS_GCP_READY`
  - `/Users/shinthiya.promi/Desktop/MS_THESIS/GCP_READY/ms_thesis_gcp_bundle.zip`
- Bundle uploaded to VM:
  - `~/ms_thesis_gcp_bundle.zip`
  - unpacked to `~/MS_THESIS_GCP_READY/implementation`
- VM environment status:
  - GPU confirmed via `nvidia-smi`: `NVIDIA L4`
  - Python 3.11 available
  - `tmux` and `unzip` installed
- NLTK resources installed manually on VM after initial failures:
  - `stopwords`
  - `punkt`
  - `punkt_tab`
  - `averaged_perceptron_tagger`
  - `averaged_perceptron_tagger_eng`
- Important runtime finding:
  - `phase2_sweep.py` should not be run with many GPU workers on a single `L4`; use CPU-only multiprocessing for the sweep and reserve GPU for single-worker `test_mpnet.py`.

## GCP Phase 2 Sweep
- Date: 2026-04-03
- Dataset: PAN 2025 validation
- Pair limit: 100
- Environment: GCP VM (`~/MS_THESIS_GCP_READY/implementation`)
- Command family used:
  - `CUDA_VISIBLE_DEVICES='' python phase2_sweep.py 100 --workers 16 --offline`
- Status at handoff:
  - run progressed to `93/100` in precompute
  - may be a heavy-tail final-pairs delay or may need rerun with fewer workers
- Resume rule:
  - if still moving, wait
  - if frozen at same count for 10-15 minutes, rerun with:
    - `CUDA_VISIBLE_DEVICES='' python phase2_sweep.py 100 --workers 4 --offline`
- Next steps after sweep:
  - inspect `phase2_results.json`
  - run `python test_mpnet.py 50 --device cuda --workers 1 --offline`
  - run final validation with winning model

## GCP Phase 2 Sweep (completed)
- Date: 2026-04-06
- Dataset: PAN 2025 validation
- Pair limit: 100
- Environment: GCP VM g2-standard-8, NVIDIA L4
- Optimisations applied:
  - Patched `phase2_sweep.py` to bypass `mp.Pool` for `--workers 1` (fixed multiprocessing deadlock at 93/100)
  - Added Numba JIT compilation for Smith-Waterman kernel (`evaluator.py`) — ~100x speedup
  - Merged Stage 1 + Stage 2 to avoid OOM (sw_cache was 32GB+ for 100 pairs)
  - Enabled GPU for precompute (removed `CUDA_VISIBLE_DEVICES=''`) — 13x faster embeddings
  - Loaded model once for workers=1 instead of per-pair reload
- Total runtime: ~64 min (precompute 15 min on GPU + SW+eval 49 min with Numba)
- Configs tested: 2160
- Baseline (Phase 1 config): P=0.3823, R=0.6587, F1=0.4712
- Best config:
  - semantic_weight: 0.95
  - structural_weight: 0.05
  - threshold: 0.70
  - gap_penalty: -0.5
  - chain_threshold: 0.1
  - min_detection_length: 300
- Best results: P=0.4898, R=0.5926, F1=0.5258
- Improvement over baseline: +11.6% F1

## Model Comparison (GCP, 50 pairs)
- Date: 2026-04-06
- Dataset: PAN 2025 validation
- Pair limit: 50
- Config: best Phase 2 config (sem=0.95, thr=0.70, gap=-0.5, chain=0.1, mindet=300)
- all-MiniLM-L6-v2: P=0.4427, R=0.5463, F1=0.4820 (477.9s)
- all-mpnet-base-v2: P=0.4014, R=0.5695, F1=0.4620 (946.4s)
- Winner: all-MiniLM-L6-v2
- Note: MiniLM is both faster (2x) and more accurate (+2pp F1) than mpnet for this task

## Experiment 5: Fine-Grained Grid Search (GCP)
- Date: 2026-04-06
- Dataset: PAN 2025 validation, 100 pairs
- Configs tested: 3024 (7 sem_weights × 8 thresholds × 3 gaps × 3 chains × 6 min_det_lengths)
- Runtime: precompute 899s (GPU) + sweep 4040s = 82 min total
- Previous best: F1=0.5258 (sem=0.95, thr=0.70)
- New best: F1=0.5263 (sem=0.94, thr=0.70, gap=-0.5, chain=0.1, mindet=300)
- Delta: +0.0005 — negligible improvement
- Conclusion: the optimum is well-converged; top 10 configs all within F1=0.5262-0.5263
- The grid confirms sem_weight 0.94-0.98 and threshold 0.70-0.72 is the optimal region

## Experiment 1: Window Size / Step Size Sweep (GCP)
- Date: 2026-04-06
- Dataset: PAN 2025 validation, 100 pairs
- Window configs tested: 9 (100/25, 120/25, 150/25, 150/50, 200/25, 200/50, 250/50, 300/50, 300/75)
- Total runtime: 2h08m
- Previous best (150/25): F1=0.5258
- New best (200/25): F1=0.5291 (P=0.4939, R=0.5956), delta=+0.0033
- Key findings:
  - Step size 25 consistently outperforms step size 50 across all window sizes
  - Larger windows (200-300) capture more context and generally improve F1
  - Window 200/25 is the sweet spot — larger windows (300) show diminishing returns
  - Full ranking: 200/25 > 150/25 > 300/50 > 250/50 > 200/50 > 300/75 > 120/25 > 150/50 > 100/25

## Experiment 3: Proper Train/Val Split (GCP)
- Date: 2026-04-06
- Train dataset: PAN 2025 train, 50 pairs
- Val dataset: PAN 2025 validation, 100 pairs
- Configs tested on train: 2160
- Best on TRAIN: F1=0.5138 (sem=0.95, thr=0.65, gap=-0.3, chain=1.0, mindet=500)
- Evaluated on VALIDATION (unseen during tuning):
  - Train-tuned config: P=0.5010, R=0.5748, **F1=0.5219**
  - Val-tuned config (old): P=0.4898, R=0.5926, **F1=0.5258**
  - Delta: -0.0038 (negligible)
- Conclusion: The model generalizes well. Only 0.4% F1 drop when tuning on separate train data. This validates the experimental methodology — the val-tuned results are not significantly overfit.
- Note: Train-tuned config has higher precision (0.50 vs 0.49) but lower recall (0.57 vs 0.59), suggesting the train data distribution slightly favours precision.
- Note: First attempt with 100 train pairs OOM'd (49GB structural matrix for a large pair). Ran with 50 train pairs after adding chunked structural matrix computation to evaluator.py.
