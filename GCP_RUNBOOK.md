# GCP Runbook

This runbook is for moving the PAN Phase 2 experiments from the local Mac to a Linux VM on Google Cloud Platform.

## What To Upload
Upload the bundled archive created from the GCP-ready copy of this project, not the original local working tree.

The bundle should contain:
- `implementation/` source files
- `implementation/data/pan25-generated-plagiarism-detection-train.zip`
- `implementation/data/pan25-generated-plagiarism-detection-validation.zip`
- `implementation/data/pan25-generated-plagiarism-detection-spot-check.zip`
- thesis tracking docs such as `EXPERIMENT_LOG.md`, `PROGRESS.md`, and `VERDICTS.md`

The bundle should not contain:
- `implementation/venv/`
- `implementation/data/extracted/`
- `.git/`
- local macOS files such as `.DS_Store`

## Recommended VM
For the remaining work, prefer one of these:

1. GPU VM for model comparison and faster embedding generation
2. High-CPU VM for large CPU-only sweeps

Reasonable starting points:
- GPU: `n1-standard-8` or better with `T4` or `L4`
- CPU: `c3-standard-22` or similar high-CPU machine

## VM Setup
After SSHing into the instance:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip unzip tmux
```

Unpack the uploaded archive:

```bash
unzip ms_thesis_gcp_bundle.zip
cd MS_THESIS_GCP_READY/implementation
```

Create the environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional but recommended for repeat runs:

```bash
mkdir -p ~/.cache/sentence-transformers
export SENTENCE_TRANSFORMERS_HOME=$HOME/.cache/sentence-transformers
```

## Run Inside Tmux
Use `tmux` so the session survives disconnects:

```bash
tmux new -s thesis
source venv/bin/activate
```

Detach with `Ctrl-b d` and reattach with:

```bash
tmux attach -t thesis
```

## Recommended Experiment Order

### 1. Re-run or expand the Phase 2 sweep
Start with the confirmed MiniLM model on more pairs:

```bash
CUDA_VISIBLE_DEVICES='' python phase2_sweep.py 100 --workers 8 --offline
```

If the VM is large enough, try more pairs:

```bash
CUDA_VISIBLE_DEVICES='' python phase2_sweep.py 200 --workers 12 --offline
```

### 2. Compare MiniLM vs mpnet with the best config
On a GPU VM, use a single worker for GPU-backed comparisons:

```bash
python test_mpnet.py 50 --device cuda --workers 1
```

If using CPU only:

```bash
python test_mpnet.py 50 --device cpu --workers 16
```

To run only the mpnet half:

```bash
python test_mpnet.py 50 --device cuda --workers 1 --only-model all-mpnet-base-v2
```

### 3. Run final validation on the winning model

```bash
python run_evaluation.py data/pan25-generated-plagiarism-detection-validation.zip 100 8 all-MiniLM-L6-v2
```

Replace the model name if mpnet wins.

## Files To Watch
- `implementation/trained_config.json`
- `implementation/phase2_results.json`
- `implementation/model_comparison.json`
- `EXPERIMENT_LOG.md`

## Practical Advice
- Do not use many multiprocessing workers with a single GPU for `test_mpnet.py`. One worker is the safer default on `cuda`.
- CPU sweeps scale well when each worker uses a single CPU thread.
- The local macOS `mps` path was unstable for mpnet multiprocessing. Linux `cuda` or Linux CPU is the intended cloud path.
- Do not run `phase2_sweep.py` with many workers on `cuda`; each worker can attach to the same GPU and cause contention. Force CPU-only sweep runs with `CUDA_VISIBLE_DEVICES=''`.
- On a fresh VM, preload the model once before offline sweeps:
  ```bash
  python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
  ```
- On a fresh VM, install NLTK resources before the first sweep:
  ```bash
  python -c "import nltk; [nltk.download(x) for x in ['stopwords','punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng']]"
  ```

## Current Handoff State (2026-04-06, all experiments done)
- VM: `instance-20260403-000831`, zone `us-central1-a`, type `g2-standard-8` with NVIDIA L4
- VM status: **STOPPED** (to save costs)
- SSH: `gcloud compute ssh instance-20260403-000831 --zone=us-central1-a`
- VM user: `shinthiya.promi` (NOT `sinthiyanawsheen`)
- Working directory: `~/MS_THESIS_GCP_READY/implementation`
- All experiments **COMPLETED**:
  - Phase 2 sweep: F1=0.5258 (100 val pairs)
  - Model comparison: MiniLM wins (F1=0.4820 vs mpnet 0.4620)
  - Exp 5 (fine grid): F1=0.5263, confirms optimum converged
  - Exp 1 (window sweep): F1=0.5291 with window 200/25 (new best)
  - Exp 3 (train/val split): F1=0.5219 on val with train-tuned config (-0.0038 drop, model generalizes well)
- All results copied to local `implementation/`:
  - `phase2_results.json`, `trained_config.json`, `model_comparison.json`
  - `exp5_fine_grid_results.json`, `exp1_window_sweep_results.json`, `exp3_train_val_results.json`
- Key code patches on VM (NOT on local machine):
  - `evaluator.py` — Numba JIT `_sw_kernel`, chunked structural matrix
  - `phase2_sweep.py` — workers=1 bypass, merged Stage 1+2
- To restart VM: `gcloud compute instances start instance-20260403-000831 --zone=us-central1-a`

## Suggested Goal
Run 3 additional experiments to improve the F1 and strengthen the thesis:

1. **Fine grid (Exp 5):** Sweep finer grid around optimum — may push F1 slightly higher
2. **Window sweep (Exp 1):** Different window/step sizes fundamentally change detection granularity
3. **Train/val split (Exp 3):** Tune on train, evaluate on val — required for methodological credibility

Current confirmed results:
- model: `all-MiniLM-L6-v2`
- best config: sem=0.95, thr=0.70, gap=-0.5, chain=0.1, mindet=300
- precision: `0.4898`
- recall: `0.5926`
- F1: `0.5258`
- validation pairs: `100`
- mpnet comparison: MiniLM wins (F1=0.4820 vs 0.4620 on 50 pairs)

## Important Runtime Notes
- Do NOT use `CUDA_VISIBLE_DEVICES=''` for single-worker runs — GPU precompute is 13x faster
- Do NOT use `mp.Pool` — use workers=1 with sequential loop to avoid deadlocks
- `evaluator.py` on the VM has Numba JIT — the local copy does NOT (patch needed if running locally)
- Fresh venv was created on VM — the old one from `sinthiyanawsheen` had broken paths
