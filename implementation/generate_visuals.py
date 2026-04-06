"""
Generate 6 thesis-quality visualizations for Chapter 4.

Usage:
    python generate_visuals.py [pair_limit]

Figures saved to implementation/figures/
"""

import os
import sys
import json
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config_utils import load_config
from data_loader import PANDataLoader
from evaluator import PANEvaluator
from features import StructuralFeatureExtractor
from scoring import HybridScorer

logging.basicConfig(level=logging.INFO, format="%(message)s")

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_ZIP = os.path.join(DATA_DIR, "pan25-generated-plagiarism-detection-train.zip")
VAL_ZIP = os.path.join(DATA_DIR, "pan25-generated-plagiarism-detection-validation.zip")

# Thesis-quality plot defaults
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.figsize": (10, 7),
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


def ensure_output_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_components():
    config = load_config()
    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    feature_extractor = StructuralFeatureExtractor()
    scorer = HybridScorer(
        semantic_weight=config["semantic_weight"],
        structural_weight=config["structural_weight"],
    )
    evaluator = PANEvaluator(
        model, feature_extractor, scorer,
        window_size=config["window_size"],
        step_size=config["step_size"],
    )
    return config, model, feature_extractor, scorer, evaluator


def find_positive_pair(loader):
    """Find a pair that has ground-truth plagiarism for visualization."""
    for susp_fn, src_fn in loader.get_pairs():
        truths = loader.load_truth(susp_fn, src_fn)
        if truths:
            return susp_fn, src_fn, truths
    return None, None, None


# ---------------------------------------------------------------------------
# Figure 1: Semantic Similarity Heatmap
# ---------------------------------------------------------------------------
def fig1_semantic_heatmap(pair_data, save_path):
    sem = pair_data["semantic_matrix"]
    # Downsample if matrix is very large for readability
    max_display = 60
    row_step = max(1, sem.shape[0] // max_display)
    col_step = max(1, sem.shape[1] // max_display)
    sem_down = sem[::row_step, ::col_step]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        sem_down, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
        xticklabels=False, yticklabels=False,
        cbar_kws={"label": "Cosine Similarity"},
    )
    ax.set_xlabel("Source Document Windows")
    ax.set_ylabel("Suspicious Document Windows")
    ax.set_title("Figure 1: Semantic Similarity Matrix (SBERT Cosine)")
    fig.savefig(save_path)
    plt.close(fig)
    logging.info("Saved: %s", save_path)


# ---------------------------------------------------------------------------
# Figure 2: Smith-Waterman Alignment Path
# ---------------------------------------------------------------------------
def fig2_alignment_path(pair_data, config, evaluator, save_path):
    sem = pair_data["semantic_matrix"]
    struct = pair_data["structural_matrix"]
    w_sem = config["semantic_weight"]
    w_str = config["structural_weight"]
    threshold = config["threshold"]
    gap_penalty = -0.5

    hybrid = w_sem * sem + w_str * struct
    m, n = hybrid.shape
    score_matrix = np.zeros((m + 1, n + 1))
    traceback = np.zeros((m + 1, n + 1), dtype=int)  # 0=zero, 1=diag, 2=up, 3=left

    # Build DP matrix with traceback
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match_score = hybrid[i - 1, j - 1]
            diag = score_matrix[i - 1, j - 1] + (match_score if match_score >= threshold else -1.0)
            up = score_matrix[i - 1, j] + gap_penalty
            left = score_matrix[i, j - 1] + gap_penalty
            best = max(0, diag, up, left)
            score_matrix[i, j] = best
            if best == 0:
                traceback[i, j] = 0
            elif best == diag:
                traceback[i, j] = 1
            elif best == up:
                traceback[i, j] = 2
            else:
                traceback[i, j] = 3

    dp_display = score_matrix[1:, 1:]

    # Traceback the optimal path from the global maximum
    path_cells = []
    # Find top-k starting points and trace back from each
    flat_scores = score_matrix[1:, 1:].flatten()
    top_starts = np.argsort(flat_scores)[::-1][:5]  # top 5 peaks
    visited = set()
    for start_idx in top_starts:
        si, sj = divmod(start_idx, n)
        ci, cj = si + 1, sj + 1  # +1 for score_matrix indexing
        while ci > 0 and cj > 0 and score_matrix[ci, cj] > 0 and (ci, cj) not in visited:
            path_cells.append((ci - 1, cj - 1))  # back to 0-indexed
            visited.add((ci, cj))
            tb = traceback[ci, cj]
            if tb == 1:
                ci -= 1; cj -= 1
            elif tb == 2:
                ci -= 1
            elif tb == 3:
                cj -= 1
            else:
                break

    # Use imshow for full-resolution display (no downsampling artifacts)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(dp_display, aspect="auto", cmap="Blues", vmin=0, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Alignment Score")

    # Overlay traceback path
    if path_cells:
        path_arr = np.array(path_cells)
        ax.scatter(path_arr[:, 1], path_arr[:, 0], c="red", s=6, alpha=0.8,
                   zorder=3, label="Traceback Path")

    ax.set_xlabel("Source Document Windows")
    ax.set_ylabel("Suspicious Document Windows")
    ax.set_title("Figure 2: Smith-Waterman Alignment Score Matrix with Traceback")
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(save_path)
    plt.close(fig)
    logging.info("Saved: %s", save_path)


# ---------------------------------------------------------------------------
# Figure 3: Hyperparameter Sweep Heatmap
# ---------------------------------------------------------------------------
def fig3_hyperparam_sweep(cached_pairs, evaluator, save_path):
    semantic_weights = np.array([0.50, 0.60, 0.70, 0.80, 0.90])
    thresholds = np.array([0.35, 0.45, 0.55, 0.65])
    f1_grid = np.zeros((len(semantic_weights), len(thresholds)))

    total = len(semantic_weights) * len(thresholds)
    progress = tqdm(total=total, desc="Sweep F1 grid", unit="config")

    for i, sw in enumerate(semantic_weights):
        stw = 1.0 - sw
        for j, thr in enumerate(thresholds):
            all_f1 = []
            for cached in cached_pairs:
                dets = evaluator.detect_plagiarism_from_precomputed(
                    cached["pair_data"],
                    threshold=thr,
                    semantic_weight=sw,
                    structural_weight=stw,
                )
                metrics = evaluator.evaluate(dets, cached["ground_truth"], cached["text_length"])
                all_f1.append(metrics["f1"])
            f1_grid[i, j] = np.mean(all_f1)
            progress.update(1)
    progress.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        f1_grid, ax=ax, cmap="YlGn", annot=True, fmt=".3f",
        xticklabels=[f"{t:.2f}" for t in thresholds],
        yticklabels=[f"{w:.2f}" for w in semantic_weights],
        cbar_kws={"label": "Macro F1 Score"},
    )
    ax.set_xlabel("Detection Threshold")
    ax.set_ylabel("Semantic Weight")
    ax.set_title("Figure 3: F1 Score Across Hyperparameter Configurations")
    fig.savefig(save_path)
    plt.close(fig)
    logging.info("Saved: %s", save_path)


# ---------------------------------------------------------------------------
# Figure 4: Precision-Recall-F1 Curve across Thresholds
# ---------------------------------------------------------------------------
def fig4_pr_curve(cached_pairs, evaluator, config, save_path):
    thresholds = np.arange(0.20, 0.80, 0.05)
    precisions, recalls, f1s = [], [], []
    sw = config["semantic_weight"]
    stw = config["structural_weight"]

    for thr in tqdm(thresholds, desc="P/R curve", unit="thr"):
        p_list, r_list, f_list = [], [], []
        for cached in cached_pairs:
            dets = evaluator.detect_plagiarism_from_precomputed(
                cached["pair_data"],
                threshold=thr,
                semantic_weight=sw,
                structural_weight=stw,
            )
            m = evaluator.evaluate(dets, cached["ground_truth"], cached["text_length"])
            p_list.append(m["precision"])
            r_list.append(m["recall"])
            f_list.append(m["f1"])
        precisions.append(np.mean(p_list))
        recalls.append(np.mean(r_list))
        f1s.append(np.mean(f_list))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precisions, "b-o", markersize=3, label="Precision")
    ax.plot(thresholds, recalls, "r-s", markersize=3, label="Recall")
    ax.plot(thresholds, f1s, "g-^", markersize=3, label="F1 Score")
    # Mark the chosen threshold
    ax.axvline(x=config["threshold"], color="gray", linestyle="--", alpha=0.7,
               label=f"Selected Threshold ({config['threshold']:.2f})")
    ax.set_xlabel("Detection Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Figure 4: Precision, Recall, and F1 vs Detection Threshold")
    ax.legend(loc="best")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path)
    plt.close(fig)
    logging.info("Saved: %s", save_path)


# ---------------------------------------------------------------------------
# Figure 5: Detection Overlay (Ground Truth vs Predicted)
# ---------------------------------------------------------------------------
def fig5_detection_overlay(susp_text, detections, truths, save_path):
    text_len = len(susp_text)
    pred_arr = np.zeros(text_len, dtype=int)
    true_arr = np.zeros(text_len, dtype=int)

    for d in detections:
        pred_arr[d["this_offset"]: d["this_offset"] + d["this_length"]] = 1
    for t in truths:
        true_arr[t["this_offset"]: t["this_offset"] + t["this_length"]] = 1

    # Build a categorical array: 0=clean, 1=TP, 2=FP, 3=FN
    overlay = np.zeros(text_len, dtype=int)
    overlay[(pred_arr == 1) & (true_arr == 1)] = 1  # TP
    overlay[(pred_arr == 1) & (true_arr == 0)] = 2  # FP
    overlay[(pred_arr == 0) & (true_arr == 1)] = 3  # FN

    # Downsample for plotting: group into bins
    bin_size = max(1, text_len // 800)
    n_bins = text_len // bin_size

    gt_bins = np.zeros(n_bins)
    pred_bins = np.zeros(n_bins)
    overlay_bins = np.zeros(n_bins)

    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size
        gt_bins[i] = true_arr[start:end].mean()
        pred_bins[i] = pred_arr[start:end].mean()
        # Dominant class in the overlay
        segment = overlay[start:end]
        counts = np.bincount(segment, minlength=4)
        if counts[1:].sum() > 0:
            overlay_bins[i] = np.argmax(counts[1:]) + 1

    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)

    # Ground truth bar
    colors_gt = ["#f0f0f0", "#2ca02c"]
    for i in range(n_bins):
        axes[0].axvspan(i, i + 1, color=colors_gt[int(gt_bins[i] > 0.5)], linewidth=0)
    axes[0].set_ylabel("Ground\nTruth", rotation=0, labelpad=50, va="center")
    axes[0].set_yticks([])
    axes[0].set_xlim(0, n_bins)

    # Prediction bar
    colors_pred = ["#f0f0f0", "#d62728"]
    for i in range(n_bins):
        axes[1].axvspan(i, i + 1, color=colors_pred[int(pred_bins[i] > 0.5)], linewidth=0)
    axes[1].set_ylabel("Model\nPrediction", rotation=0, labelpad=50, va="center")
    axes[1].set_yticks([])

    # Overlay bar (TP/FP/FN)
    overlay_colors = {0: "#f0f0f0", 1: "#2ca02c", 2: "#d62728", 3: "#ff7f0e"}
    for i in range(n_bins):
        axes[2].axvspan(i, i + 1, color=overlay_colors[int(overlay_bins[i])], linewidth=0)
    axes[2].set_ylabel("Overlap\nAnalysis", rotation=0, labelpad=50, va="center")
    axes[2].set_yticks([])
    axes[2].set_xlabel("Document Position (characters)")

    # Custom legend for bottom
    legend_patches = [
        mpatches.Patch(color="#2ca02c", label="True Positive"),
        mpatches.Patch(color="#d62728", label="False Positive"),
        mpatches.Patch(color="#ff7f0e", label="False Negative"),
        mpatches.Patch(color="#f0f0f0", label="Clean / No Detection"),
    ]
    axes[2].legend(handles=legend_patches, loc="upper center",
                   bbox_to_anchor=(0.5, -0.35), ncol=4, fontsize=9)

    fig.suptitle("Figure 5: Character-Level Detection Overlay", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(save_path)
    plt.close(fig)
    logging.info("Saved: %s", save_path)


# ---------------------------------------------------------------------------
# Figure 6: Structural Feature Contribution (Hybrid vs Semantic-Only)
# ---------------------------------------------------------------------------
def fig6_structural_contribution(cached_pairs, evaluator, config, save_path):
    thresholds = np.array([0.30, 0.40, 0.50, 0.55, 0.60, 0.70])
    hybrid_f1s = []
    semantic_only_f1s = []

    sw = config["semantic_weight"]
    stw = config["structural_weight"]

    for thr in tqdm(thresholds, desc="Feature contribution", unit="thr"):
        h_f1_list, s_f1_list = [], []
        for cached in cached_pairs:
            # Hybrid
            dets_h = evaluator.detect_plagiarism_from_precomputed(
                cached["pair_data"], threshold=thr,
                semantic_weight=sw, structural_weight=stw,
            )
            m_h = evaluator.evaluate(dets_h, cached["ground_truth"], cached["text_length"])
            h_f1_list.append(m_h["f1"])

            # Semantic only
            dets_s = evaluator.detect_plagiarism_from_precomputed(
                cached["pair_data"], threshold=thr,
                semantic_weight=1.0, structural_weight=0.0,
            )
            m_s = evaluator.evaluate(dets_s, cached["ground_truth"], cached["text_length"])
            s_f1_list.append(m_s["f1"])

        hybrid_f1s.append(np.mean(h_f1_list))
        semantic_only_f1s.append(np.mean(s_f1_list))

    x = np.arange(len(thresholds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, hybrid_f1s, width, label=f"Hybrid (sem={sw:.1f}, str={stw:.1f})", color="#2ca02c")
    bars2 = ax.bar(x + width / 2, semantic_only_f1s, width, label="Semantic Only (sem=1.0)", color="#1f77b4")
    ax.set_xlabel("Detection Threshold")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("Figure 6: Hybrid Model vs Semantic-Only Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(hybrid_f1s), max(semantic_only_f1s)) * 1.15 + 0.01)
    fig.savefig(save_path)
    plt.close(fig)
    logging.info("Saved: %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pair_limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    ensure_output_dir()

    logging.info("Loading model and config...")
    config, model, feature_extractor, scorer, evaluator = load_components()
    logging.info("Config: %s", json.dumps({k: v for k, v in config.items()
                                            if k in ("semantic_weight", "structural_weight",
                                                      "threshold", "window_size", "step_size")}))

    # --- Load validation data ---
    logging.info("Loading PAN validation data (limit=%s)...", pair_limit)
    loader = PANDataLoader(VAL_ZIP)
    all_pairs = loader.get_pairs()

    # Find a positive pair for single-pair visuals (Figs 1, 2, 5)
    logging.info("Searching for a positive (plagiarised) pair...")
    pos_susp, pos_src, pos_truths = find_positive_pair(loader)
    if pos_susp is None:
        logging.warning("No positive pair found in validation set; trying train set.")
        loader_train = PANDataLoader(TRAIN_ZIP)
        pos_susp, pos_src, pos_truths = find_positive_pair(loader_train)
        loader_for_pos = loader_train
    else:
        loader_for_pos = loader

    if pos_susp:
        logging.info("Using positive pair: %s vs %s (%d truth spans)", pos_susp, pos_src, len(pos_truths))
        susp_text = loader_for_pos.load_text(pos_susp, is_suspicious=True)
        src_text = loader_for_pos.load_text(pos_src, is_suspicious=False)
        pair_data = evaluator.precompute_pair_data(susp_text, src_text)

        # Figure 1: Semantic Heatmap
        fig1_semantic_heatmap(pair_data, os.path.join(FIGURES_DIR, "fig1_semantic_heatmap.png"))

        # Figure 2: Alignment Path
        fig2_alignment_path(pair_data, config, evaluator, os.path.join(FIGURES_DIR, "fig2_alignment_path.png"))

        # Figure 5: Detection Overlay (needs detections)
        detections = evaluator.detect_plagiarism_from_precomputed(
            pair_data, threshold=config["threshold"],
            semantic_weight=config["semantic_weight"],
            structural_weight=config["structural_weight"],
        )
        fig5_detection_overlay(susp_text, detections, pos_truths,
                               os.path.join(FIGURES_DIR, "fig5_detection_overlay.png"))
    else:
        logging.error("Could not find any positive pair. Skipping Figs 1, 2, 5.")

    # --- Build cache for multi-pair figures (Figs 3, 4, 6) ---
    subset = all_pairs[:pair_limit]
    logging.info("Precomputing %d validation pairs for sweep figures...", len(subset))
    cached_pairs = []
    for susp_fn, src_fn in tqdm(subset, desc="Precomputing pairs", unit="pair"):
        s_text = loader.load_text(susp_fn, is_suspicious=True)
        sr_text = loader.load_text(src_fn, is_suspicious=False)
        gt = loader.load_truth(susp_fn, src_fn)
        pd = evaluator.precompute_pair_data(s_text, sr_text)
        cached_pairs.append({
            "pair_data": pd,
            "ground_truth": gt,
            "text_length": len(s_text),
        })

    # Figure 3: Hyperparameter Sweep
    fig3_hyperparam_sweep(cached_pairs, evaluator, os.path.join(FIGURES_DIR, "fig3_hyperparam_sweep.png"))

    # Figure 4: P/R/F1 Curve
    fig4_pr_curve(cached_pairs, evaluator, config, os.path.join(FIGURES_DIR, "fig4_pr_curve.png"))

    # Figure 6: Structural Contribution
    fig6_structural_contribution(cached_pairs, evaluator, config,
                                  os.path.join(FIGURES_DIR, "fig6_structural_contribution.png"))

    logging.info("All figures saved to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
