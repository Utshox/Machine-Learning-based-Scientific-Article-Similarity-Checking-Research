"""Experiment 3: Proper train/val split — tune on train, evaluate on validation.

This addresses the methodological weakness of tuning and evaluating on the same data.
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

TRAIN_DATASET = "pan25-generated-plagiarism-detection-train.zip"
VAL_DATASET = "pan25-generated-plagiarism-detection-validation.zip"


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


def precompute_pairs(model, pairs, dataset_path, window_size=150, step_size=25):
    fe = StructuralFeatureExtractor()
    hs = HybridScorer(semantic_weight=0.7, structural_weight=0.3)
    ev = PANEvaluator(model, fe, hs, window_size=window_size, step_size=step_size)
    ld = PANDataLoader(dataset_path)

    cached = []
    for susp_fn, src_fn in tqdm(pairs, desc="Precomputing", unit="pair"):
        susp_text = ld.load_text(susp_fn, is_suspicious=True)
        src_text = ld.load_text(src_fn, is_suspicious=False)
        gt = ld.load_truth(susp_fn, src_fn)
        pd = ev.precompute_pair_data(susp_text, src_text)
        cached.append({
            "ground_truth": gt, "pair_data": pd,
            "text_length": len(susp_text),
        })
    return cached, ev


def sweep_configs(cached_pairs, evaluator):
    """Run full hyperparameter sweep, return best config and top10."""
    semantic_weights = [0.85, 0.90, 0.92, 0.95, 0.98]
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    gap_penalties = [-0.3, -0.5, -0.7]
    chain_thresholds = [0.1, 0.5, 1.0, 2.0]
    min_det_lengths = [0, 100, 200, 300, 400, 500]

    sw_combos = list(product(semantic_weights, thresholds, gap_penalties))
    post_combos = list(product(chain_thresholds, min_det_lengths))
    total = len(sw_combos) * len(post_combos)

    best_result = None
    top10 = []

    for sw, thr, gp in tqdm(sw_combos, desc="SW+eval", unit="combo"):
        pair_raw_dets = []
        for cached in cached_pairs:
            raw = get_raw_detections(evaluator, cached["pair_data"], thr, gp, sw)
            pair_raw_dets.append(raw)

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
            top10.append(result)
            top10.sort(key=lambda x: x["f1"], reverse=True)
            top10 = top10[:10]
            if best_result is None or result["f1"] > best_result["f1"]:
                best_result = result

        del pair_raw_dets

    return best_result, top10, total


def evaluate_config(cached_pairs, evaluator, config):
    """Evaluate a single config on cached pairs."""
    precisions, recalls, f1s = [], [], []
    for cached in cached_pairs:
        raw = get_raw_detections(evaluator, cached["pair_data"],
                                 config["threshold"], config["gap_penalty"],
                                 config["semantic_weight"])
        dets = apply_post_filters(raw, config["chain_threshold"],
                                   config["min_detection_length"], evaluator)
        m = evaluator.evaluate(dets, cached["ground_truth"], cached["text_length"])
        precisions.append(m["precision"])
        recalls.append(m["recall"])
        f1s.append(m["f1"])

    n = len(cached_pairs)
    return {
        "precision": sum(precisions) / n,
        "recall": sum(recalls) / n,
        "f1": sum(f1s) / n,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pairs", type=int, default=100)
    parser.add_argument("--val-pairs", type=int, default=100)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    train_path = resolve_dataset_path(None, TRAIN_DATASET)
    val_path = resolve_dataset_path(None, VAL_DATASET)

    logging.info("=" * 60)
    logging.info("EXPERIMENT 3: PROPER TRAIN/VAL SPLIT")
    logging.info(f"Train pairs: {args.train_pairs}")
    logging.info(f"Val pairs: {args.val_pairs}")
    logging.info("=" * 60)

    model = load_sentence_transformer("all-MiniLM-L6-v2", offline=args.offline)

    # --- STAGE 1: Tune on TRAIN data ---
    logging.info("\n--- STAGE 1: Tuning on TRAIN split ---")
    train_loader = PANDataLoader(train_path)
    train_pairs = train_loader.get_pairs()[:args.train_pairs]
    logging.info(f"Train pairs loaded: {len(train_pairs)}")

    t0 = time.time()
    train_cached, train_ev = precompute_pairs(model, train_pairs, train_path)
    t_train_precompute = time.time() - t0
    logging.info(f"Train precompute: {t_train_precompute:.1f}s")

    t0 = time.time()
    best_train, top10_train, n_configs = sweep_configs(train_cached, train_ev)
    t_train_sweep = time.time() - t0
    logging.info(f"Train sweep: {t_train_sweep:.1f}s ({n_configs} configs)")
    logging.info(f"Best on TRAIN: P={best_train['precision']:.4f} "
                 f"R={best_train['recall']:.4f} F1={best_train['f1']:.4f}")
    logging.info(f"Config: sem={best_train['semantic_weight']}, "
                 f"thr={best_train['threshold']}, gap={best_train['gap_penalty']}, "
                 f"chain={best_train['chain_threshold']}, mindet={best_train['min_detection_length']}")

    # Free train data
    del train_cached
    import gc; gc.collect()

    # --- STAGE 2: Evaluate best config on VALIDATION data ---
    logging.info("\n--- STAGE 2: Evaluating on VALIDATION split ---")
    val_loader = PANDataLoader(val_path)
    val_pairs = val_loader.get_pairs()[:args.val_pairs]
    logging.info(f"Val pairs loaded: {len(val_pairs)}")

    t0 = time.time()
    val_cached, val_ev = precompute_pairs(model, val_pairs, val_path)
    t_val_precompute = time.time() - t0
    logging.info(f"Val precompute: {t_val_precompute:.1f}s")

    val_result = evaluate_config(val_cached, val_ev, best_train)
    logging.info(f"\nResults on VALIDATION (unseen during tuning):")
    logging.info(f"  Precision: {val_result['precision']:.4f}")
    logging.info(f"  Recall:    {val_result['recall']:.4f}")
    logging.info(f"  F1:        {val_result['f1']:.4f}")

    # Also evaluate the old config on validation for comparison
    old_config = {
        "semantic_weight": 0.95, "threshold": 0.70, "gap_penalty": -0.5,
        "chain_threshold": 0.1, "min_detection_length": 300,
    }
    old_val_result = evaluate_config(val_cached, val_ev, old_config)
    logging.info(f"\nOld config (tuned on val) evaluated on VALIDATION:")
    logging.info(f"  Precision: {old_val_result['precision']:.4f}")
    logging.info(f"  Recall:    {old_val_result['recall']:.4f}")
    logging.info(f"  F1:        {old_val_result['f1']:.4f}")

    # Report
    logging.info(f"\n{'='*60}")
    logging.info("SUMMARY")
    logging.info(f"{'='*60}")
    logging.info(f"Train-tuned config on val:  F1={val_result['f1']:.4f}")
    logging.info(f"Val-tuned config on val:    F1={old_val_result['f1']:.4f}")
    logging.info(f"Difference: {val_result['f1'] - old_val_result['f1']:+.4f}")

    # Save
    results_path = os.path.join(os.path.dirname(__file__), "exp3_train_val_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "train_pairs": args.train_pairs,
            "val_pairs": args.val_pairs,
            "configs_tested": n_configs,
            "best_train_config": best_train,
            "best_train_top10": top10_train,
            "train_precompute_time": t_train_precompute,
            "train_sweep_time": t_train_sweep,
            "val_precompute_time": t_val_precompute,
            "val_result_train_tuned": val_result,
            "val_result_val_tuned": old_val_result,
        }, f, indent=2)
    logging.info(f"Saved to {results_path}")

    append_experiment_log("Experiment 3: Train/Val Split", {
        "train_pairs": args.train_pairs,
        "val_pairs": args.val_pairs,
        "configs_tested": n_configs,
        "best_train_f1": best_train["f1"],
        "val_f1_train_tuned": val_result["f1"],
        "val_f1_val_tuned": old_val_result["f1"],
        "best_train_config": best_train,
    })


if __name__ == "__main__":
    main()
