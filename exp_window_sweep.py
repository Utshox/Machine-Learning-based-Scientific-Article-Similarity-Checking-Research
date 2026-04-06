"""Experiment 1: Window size / step size sweep.

Tests different window_size and step_size combinations to find the optimal
granularity for plagiarism detection. Each combo requires a full precompute
since window parameters change the embedding matrices.
"""
import json
import time
import logging
import os

import numpy as np
from itertools import product
from tqdm import tqdm

from data_loader import PANDataLoader
from evaluator import PANEvaluator
from features import StructuralFeatureExtractor
from runtime_utils import load_sentence_transformer, resolve_dataset_path
from scoring import HybridScorer
from experiment_logger import append_experiment_log

logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DATASET = "pan25-generated-plagiarism-detection-validation.zip"


def get_raw_detections(evaluator, pair_data, threshold, gap_penalty, semantic_weight):
    if pair_data is None:
        return []
    sem_w = semantic_weight
    str_w = 1.0 - semantic_weight
    hybrid_matrix = (sem_w * pair_data["semantic_matrix"] +
                     str_w * pair_data["structural_matrix"])
    score_matrix = evaluator._smith_waterman_fast(hybrid_matrix, threshold, gap_penalty)
    m, n = hybrid_matrix.shape
    susp_windows = pair_data["susp_windows"]
    src_windows = pair_data["src_windows"]
    scores = score_matrix[1:, 1:]
    ge_up = scores >= score_matrix[:m, 1:]
    ge_left = scores >= score_matrix[1:, :n]
    mask = (scores > 0) & ge_up & ge_left
    det_i, det_j = np.where(mask)
    raw_dets = []
    for idx in range(len(det_i)):
        i, j = det_i[idx], det_j[idx]
        raw_dets.append({
            "this_offset": susp_windows[i]["offset"],
            "this_length": susp_windows[i]["length"],
            "source_offset": src_windows[j]["offset"],
            "source_length": src_windows[j]["length"],
            "score": float(scores[i, j]),
        })
    return raw_dets


def apply_post_filters(raw_dets, chain_threshold, min_detection_length, evaluator):
    filtered = [d for d in raw_dets if d["score"] > chain_threshold]
    merged = evaluator.merge_detections(filtered)
    if min_detection_length > 0:
        merged = [d for d in merged if d["this_length"] >= min_detection_length]
    return merged


def evaluate_window_config(model, pairs, dataset_path, window_size, step_size):
    """Precompute + sweep for one window/step combo."""
    fe = StructuralFeatureExtractor()
    hs = HybridScorer(semantic_weight=0.7, structural_weight=0.3)
    ev = PANEvaluator(model, fe, hs, window_size=window_size, step_size=step_size)
    ld = PANDataLoader(dataset_path)

    # Precompute with this window config
    cached_pairs = []
    for susp_fn, src_fn in pairs:
        susp_text = ld.load_text(susp_fn, is_suspicious=True)
        src_text = ld.load_text(src_fn, is_suspicious=False)
        gt = ld.load_truth(susp_fn, src_fn)
        pd = ev.precompute_pair_data(susp_text, src_text)
        cached_pairs.append({
            "ground_truth": gt, "pair_data": pd,
            "text_length": len(susp_text),
        })

    # Sweep core params for this window config
    semantic_weights = [0.90, 0.95, 0.98]
    thresholds = [0.60, 0.65, 0.70, 0.75]
    gap_penalties = [-0.5]
    chain_thresholds = [0.1, 1.0]
    min_det_lengths = [200, 300, 400]

    sw_combos = list(product(semantic_weights, thresholds, gap_penalties))
    post_combos = list(product(chain_thresholds, min_det_lengths))

    best_result = None
    for sw, thr, gp in sw_combos:
        pair_raw_dets = []
        for cached in cached_pairs:
            raw = get_raw_detections(ev, cached["pair_data"], thr, gp, sw)
            pair_raw_dets.append(raw)

        for ct, mdl in post_combos:
            precisions, recalls, f1s = [], [], []
            for pair_idx, cached in enumerate(cached_pairs):
                dets = apply_post_filters(pair_raw_dets[pair_idx], ct, mdl, ev)
                m = ev.evaluate(dets, cached["ground_truth"], cached["text_length"])
                precisions.append(m["precision"])
                recalls.append(m["recall"])
                f1s.append(m["f1"])

            n = len(cached_pairs)
            result = {
                "window_size": window_size,
                "step_size": step_size,
                "semantic_weight": sw,
                "threshold": thr,
                "gap_penalty": gp,
                "chain_threshold": ct,
                "min_detection_length": mdl,
                "precision": sum(precisions) / n,
                "recall": sum(recalls) / n,
                "f1": sum(f1s) / n,
            }
            if best_result is None or result["f1"] > best_result["f1"]:
                best_result = result

        del pair_raw_dets

    del cached_pairs
    return best_result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pair_limit", nargs="?", type=int, default=100)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    dataset_path = resolve_dataset_path(args.dataset, DEFAULT_DATASET)
    pair_limit = args.pair_limit

    logging.info("=" * 60)
    logging.info("EXPERIMENT 1: WINDOW SIZE / STEP SIZE SWEEP")
    logging.info(f"Pairs: {pair_limit}")
    logging.info("=" * 60)

    model = load_sentence_transformer("all-MiniLM-L6-v2", offline=args.offline)
    ld = PANDataLoader(dataset_path)
    pairs = ld.get_pairs()[:pair_limit]

    # Window configs to try (avoid very small windows — too many windows per doc causes OOM)
    window_configs = [
        (100, 25),
        (120, 25),
        (150, 25),   # current default
        (150, 50),   # same window, less overlap
        (200, 25),
        (200, 50),
        (250, 50),
        (300, 50),
        (300, 75),
    ]

    all_results = []
    overall_best = None

    for ws, ss in tqdm(window_configs, desc="Window configs", unit="config"):
        logging.info(f"\n--- Window: {ws}, Step: {ss} ---")
        t0 = time.time()
        best = evaluate_window_config(model, pairs, dataset_path, ws, ss)
        elapsed = time.time() - t0
        best["time"] = elapsed
        all_results.append(best)
        logging.info(f"  Best F1={best['f1']:.4f} (P={best['precision']:.4f} R={best['recall']:.4f}) "
                     f"[{elapsed:.0f}s]")

        if overall_best is None or best["f1"] > overall_best["f1"]:
            overall_best = best

    # Report
    logging.info(f"\n{'='*80}")
    logging.info("WINDOW SWEEP RESULTS")
    logging.info(f"{'='*80}")
    logging.info(f"{'Win':<6} {'Step':<6} {'SemW':<6} {'Thr':<6} {'Chain':<6} "
                 f"{'MinL':<6} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Time':<8}")
    logging.info("-" * 80)

    all_results.sort(key=lambda x: x["f1"], reverse=True)
    for r in all_results:
        logging.info(
            f"{r['window_size']:<6} {r['step_size']:<6} {r['semantic_weight']:<6.2f} "
            f"{r['threshold']:<6.2f} {r['chain_threshold']:<6.1f} "
            f"{r['min_detection_length']:<6} {r['precision']:<8.4f} "
            f"{r['recall']:<8.4f} {r['f1']:<8.4f} {r.get('time',0):<8.0f}")

    logging.info(f"\nOVERALL BEST: window={overall_best['window_size']}, step={overall_best['step_size']}")
    logging.info(f"  F1={overall_best['f1']:.4f} (P={overall_best['precision']:.4f} R={overall_best['recall']:.4f})")
    logging.info(f"  Previous best (150/25): F1=0.5258")
    logging.info(f"  Delta: {overall_best['f1'] - 0.5258:+.4f}")

    # Save
    results_path = os.path.join(os.path.dirname(__file__), "exp1_window_sweep_results.json")
    with open(results_path, "w") as f:
        json.dump({"pair_limit": pair_limit, "window_configs_tested": len(window_configs),
                    "overall_best": overall_best, "all_results": all_results}, f, indent=2)
    logging.info(f"Saved to {results_path}")

    append_experiment_log("Experiment 1: Window Sweep", {
        "dataset": dataset_path, "pair_limit": pair_limit,
        "window_configs_tested": len(window_configs),
        "best_window_size": overall_best["window_size"],
        "best_step_size": overall_best["step_size"],
        "best_f1": overall_best["f1"],
        "best_config": overall_best,
        "previous_best_f1": 0.5258,
    })


if __name__ == "__main__":
    main()
