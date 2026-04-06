"""
Phase 2 Hyperparameter Sweep — Optimised two-stage approach.

Stage 1: Run Smith-Waterman once per (threshold, gap_penalty, semantic_weight) combo,
         store raw detections. This is the expensive part.
Stage 2: Sweep chain_threshold and min_detection_length over cached raw detections.
         This is instant (no DP recomputation).

Usage:
    python phase2_sweep.py [pair_limit]   (default: 50)
"""

import os
import json
import time
import logging
import multiprocessing as mp
import argparse

import numpy as np
from itertools import product
from tqdm import tqdm

from config_utils import save_config
from data_loader import PANDataLoader
from evaluator import PANEvaluator
from experiment_logger import append_experiment_log
from features import StructuralFeatureExtractor
from runtime_utils import configure_cpu_runtime, load_sentence_transformer, resolve_dataset_path
from scoring import HybridScorer

logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DATASET = "pan25-generated-plagiarism-detection-validation.zip"


def _precompute_worker(args):
    """Worker: precompute embeddings + features for one pair."""
    susp_fn, src_fn, dataset_path, window_size, step_size, model_name, offline = args
    configure_cpu_runtime(1)
    model = load_sentence_transformer(model_name, offline=offline)
    fe = StructuralFeatureExtractor()
    hs = HybridScorer(semantic_weight=0.7, structural_weight=0.3)
    ev = PANEvaluator(model, fe, hs, window_size=window_size, step_size=step_size)
    ld = PANDataLoader(dataset_path)

    susp_text = ld.load_text(susp_fn, is_suspicious=True)
    src_text = ld.load_text(src_fn, is_suspicious=False)
    ground_truth = ld.load_truth(susp_fn, src_fn)
    pair_data = ev.precompute_pair_data(susp_text, src_text)

    return {
        "susp_fn": susp_fn,
        "src_fn": src_fn,
        "ground_truth": ground_truth,
        "pair_data": pair_data,
        "text_length": len(susp_text),
    }


def get_raw_detections(evaluator, pair_data, threshold, gap_penalty, semantic_weight):
    """
    Run Smith-Waterman and return ALL raw detections (before chain/length filtering).
    Each detection includes its score so we can filter by chain_threshold later.
    """
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

    # Extract ALL cells above zero (we'll filter by chain_threshold later)
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
    """Apply chain_threshold and min_detection_length to raw detections — instant."""
    filtered = [d for d in raw_dets if d["score"] > chain_threshold]
    merged = evaluator.merge_detections(filtered)
    if min_detection_length > 0:
        merged = [d for d in merged if d["this_length"] >= min_detection_length]
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pair_limit", nargs="?", type=int, default=50)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    pair_limit = args.pair_limit
    n_workers = args.workers
    dataset_path = resolve_dataset_path(args.dataset, DEFAULT_DATASET)

    logging.info("=" * 60)
    logging.info("PHASE 2 HYPERPARAMETER SWEEP (optimised two-stage)")
    logging.info(f"Validation pairs: {pair_limit}")
    logging.info(f"CPU cores: {n_workers}")
    logging.info("=" * 60)

    # --- Stage 0: Parallel precomputation ---
    loader = PANDataLoader(dataset_path)
    all_pairs = loader.get_pairs()[:pair_limit]

    work_items = [
        (susp_fn, src_fn, dataset_path, 150, 25, args.model, args.offline)
        for susp_fn, src_fn in all_pairs
    ]

    logging.info(f"Precomputing {len(all_pairs)} pairs across {n_workers} cores...")
    t0 = time.time()
    cached_pairs = []
    with mp.Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap(_precompute_worker, work_items),
            total=len(work_items),
            desc="Precomputing",
            unit="pair",
        ):
            cached_pairs.append(result)
    t_precompute = time.time() - t0
    logging.info(f"Precomputation: {t_precompute:.1f}s ({t_precompute/len(cached_pairs):.1f}s/pair)")

    # --- Stage 1: Run SW for each (threshold, gap_penalty, semantic_weight) ---
    # These are the EXPENSIVE params (each needs a full DP pass)
    semantic_weights = [0.80, 0.85, 0.90, 0.95]
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    gap_penalties = [-0.5, -1.0, -1.5]

    sw_combos = list(product(semantic_weights, thresholds, gap_penalties))
    logging.info(f"\nStage 1: Running Smith-Waterman for {len(sw_combos)} "
                 f"(sem_w x thr x gap) combos on {len(cached_pairs)} pairs...")

    configure_cpu_runtime(max(1, min(4, os.cpu_count() or 1)))
    model = load_sentence_transformer(args.model, offline=args.offline)
    fe = StructuralFeatureExtractor()
    hs = HybridScorer(semantic_weight=0.8, structural_weight=0.2)
    evaluator = PANEvaluator(model, fe, hs, window_size=150, step_size=25)

    # Cache: key=(sw, thr, gp) -> list of raw_dets per pair
    sw_cache = {}
    t1 = time.time()
    for sw, thr, gp in tqdm(sw_combos, desc="SW combos", unit="combo"):
        pair_raw_dets = []
        for cached in cached_pairs:
            raw = get_raw_detections(evaluator, cached["pair_data"], thr, gp, sw)
            pair_raw_dets.append(raw)
        sw_cache[(sw, thr, gp)] = pair_raw_dets
    t_sw = time.time() - t1
    logging.info(f"Stage 1: {t_sw:.1f}s ({t_sw/len(sw_combos):.2f}s/combo)")

    # --- Stage 2: Sweep post-processing params (FREE — no DP) ---
    chain_thresholds = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
    min_det_lengths = [0, 100, 200, 300, 500]

    post_combos = list(product(chain_thresholds, min_det_lengths))
    total_configs = len(sw_combos) * len(post_combos)
    logging.info(f"\nStage 2: Sweeping {len(post_combos)} post-processing combos "
                 f"x {len(sw_combos)} SW combos = {total_configs} total configs (instant)...")

    best_result = None
    all_results = []
    top10 = []

    t2 = time.time()
    for (sw, thr, gp), pair_raw_dets in tqdm(sw_cache.items(), desc="Evaluating", unit="combo"):
        for ct, mdl in post_combos:
            precisions, recalls, f1s = [], [], []
            for pair_idx, cached in enumerate(cached_pairs):
                dets = apply_post_filters(pair_raw_dets[pair_idx], ct, mdl, evaluator)
                m = evaluator.evaluate(dets, cached["ground_truth"], cached["text_length"])
                precisions.append(m["precision"])
                recalls.append(m["recall"])
                f1s.append(m["f1"])

            n = len(cached_pairs)
            result = {
                "semantic_weight": sw,
                "structural_weight": round(1.0 - sw, 4),
                "threshold": thr,
                "gap_penalty": gp,
                "chain_threshold": ct,
                "min_detection_length": mdl,
                "precision": sum(precisions) / n,
                "recall": sum(recalls) / n,
                "f1": sum(f1s) / n,
            }
            all_results.append(result)

            top10.append(result)
            top10.sort(key=lambda x: x["f1"], reverse=True)
            top10 = top10[:10]

            if best_result is None or result["f1"] > best_result["f1"]:
                best_result = result

    t_post = time.time() - t2

    # --- Report ---
    logging.info("\n" + "=" * 70)
    logging.info("PHASE 2 RESULTS")
    logging.info("=" * 70)
    logging.info(f"Configs tested: {total_configs}")
    logging.info(f"Precompute: {t_precompute:.1f}s | SW stage: {t_sw:.1f}s | "
                 f"Post-filter: {t_post:.1f}s | Total: {t_precompute+t_sw+t_post:.1f}s")

    logging.info(f"\n{'Rank':<5} {'SemW':<6} {'Thr':<6} {'Gap':<6} "
                 f"{'Chain':<6} {'MinL':<6} {'Prec':<8} {'Rec':<8} {'F1':<8}")
    logging.info("-" * 72)
    for rank, r in enumerate(top10, 1):
        logging.info(
            f"{rank:<5} {r['semantic_weight']:<6.2f} {r['threshold']:<6.2f} "
            f"{r['gap_penalty']:<6.1f} {r['chain_threshold']:<6.1f} "
            f"{r['min_detection_length']:<6} {r['precision']:<8.4f} "
            f"{r['recall']:<8.4f} {r['f1']:<8.4f}"
        )

    # Baseline comparison
    baseline_result = None
    for r in all_results:
        if (r["semantic_weight"] == 0.80 and r["threshold"] == 0.55 and
            r["gap_penalty"] == -0.5 and r["chain_threshold"] == 0.1 and
            r["min_detection_length"] == 0):
            baseline_result = r
            break

    if baseline_result:
        logging.info(f"\nBASELINE (Phase 1): P={baseline_result['precision']:.4f} "
                     f"R={baseline_result['recall']:.4f} F1={baseline_result['f1']:.4f}")
    logging.info(f"BEST    (Phase 2): P={best_result['precision']:.4f} "
                 f"R={best_result['recall']:.4f} F1={best_result['f1']:.4f}")
    if baseline_result:
        improvement = best_result["f1"] - baseline_result["f1"]
        logging.info(f"IMPROVEMENT: +{improvement:.4f} F1 "
                     f"({improvement/max(baseline_result['f1'],0.001)*100:.1f}%)")

    # Save best config
    best_save = {
        "semantic_weight": best_result["semantic_weight"],
        "structural_weight": best_result["structural_weight"],
        "threshold": best_result["threshold"],
        "chain_threshold": best_result["chain_threshold"],
        "gap_penalty": best_result["gap_penalty"],
        "min_detection_length": best_result["min_detection_length"],
        "window_size": 150,
        "step_size": 25,
        "precision": best_result["precision"],
        "recall": best_result["recall"],
        "f1": best_result["f1"],
    }
    save_config(best_save)
    logging.info(f"\nSaved best config to trained_config.json")

    # Save full results
    results_path = os.path.join(os.path.dirname(__file__), "phase2_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "pair_limit": pair_limit,
            "total_configs": total_configs,
            "precompute_time": t_precompute,
            "sw_time": t_sw,
            "post_time": t_post,
            "dataset": dataset_path,
            "model_name": args.model,
            "baseline": baseline_result,
            "best": best_result,
            "top10": top10,
        }, f, indent=2)
    logging.info(f"Results saved to {results_path}")

    append_experiment_log(
        "Phase 2 Sweep",
        {
            "dataset": dataset_path,
            "pair_limit": pair_limit,
            "configs_tested": total_configs,
            "model_name": args.model,
            "baseline_f1": baseline_result["f1"] if baseline_result else "N/A",
            "best_semantic_weight": best_result["semantic_weight"],
            "best_threshold": best_result["threshold"],
            "best_chain_threshold": best_result["chain_threshold"],
            "best_gap_penalty": best_result["gap_penalty"],
            "best_min_detection_length": best_result["min_detection_length"],
            "best_precision": best_result["precision"],
            "best_recall": best_result["recall"],
            "best_f1": best_result["f1"],
        },
    )


if __name__ == "__main__":
    main()
