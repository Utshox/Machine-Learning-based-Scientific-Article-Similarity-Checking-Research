"""Experiment 5: Finer hyperparameter grid around the current optimum."""
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
    logging.info("EXPERIMENT 5: FINE-GRAINED GRID SEARCH")
    logging.info(f"Pairs: {pair_limit}")
    logging.info("=" * 60)

    # Load model once
    model = load_sentence_transformer("all-MiniLM-L6-v2", offline=args.offline)
    fe = StructuralFeatureExtractor()
    hs = HybridScorer(semantic_weight=0.7, structural_weight=0.3)
    ev = PANEvaluator(model, fe, hs, window_size=150, step_size=25)
    ld = PANDataLoader(dataset_path)

    pairs = ld.get_pairs()[:pair_limit]

    # Precompute
    logging.info(f"Precomputing {len(pairs)} pairs...")
    t0 = time.time()
    cached_pairs = []
    for susp_fn, src_fn in tqdm(pairs, desc="Precomputing", unit="pair"):
        susp_text = ld.load_text(susp_fn, is_suspicious=True)
        src_text = ld.load_text(src_fn, is_suspicious=False)
        gt = ld.load_truth(susp_fn, src_fn)
        pd = ev.precompute_pair_data(susp_text, src_text)
        cached_pairs.append({
            "susp_fn": susp_fn, "src_fn": src_fn,
            "ground_truth": gt, "pair_data": pd,
            "text_length": len(susp_text),
        })
    t_precompute = time.time() - t0
    logging.info(f"Precompute: {t_precompute:.1f}s")

    # Fine grid around optimum
    semantic_weights = [0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 1.00]
    thresholds = [0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80]
    gap_penalties = [-0.3, -0.5, -0.7]
    chain_thresholds = [0.1, 0.5, 1.0]
    min_det_lengths = [200, 250, 300, 350, 400, 500]

    sw_combos = list(product(semantic_weights, thresholds, gap_penalties))
    post_combos = list(product(chain_thresholds, min_det_lengths))
    total_configs = len(sw_combos) * len(post_combos)
    logging.info(f"\nSweeping {len(sw_combos)} SW combos x {len(post_combos)} post = {total_configs} configs")

    best_result = None
    top10 = []

    t1 = time.time()
    for sw, thr, gp in tqdm(sw_combos, desc="SW+eval", unit="combo"):
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

    t_sweep = time.time() - t1

    logging.info(f"\n{'='*70}")
    logging.info("FINE GRID RESULTS")
    logging.info(f"{'='*70}")
    logging.info(f"Configs tested: {total_configs}")
    logging.info(f"Time: precompute {t_precompute:.1f}s + sweep {t_sweep:.1f}s")
    logging.info(f"\n{'Rank':<5} {'SemW':<6} {'Thr':<6} {'Gap':<6} "
                 f"{'Chain':<6} {'MinL':<6} {'Prec':<8} {'Rec':<8} {'F1':<8}")
    logging.info("-" * 72)
    for rank, r in enumerate(top10, 1):
        logging.info(
            f"{rank:<5} {r['semantic_weight']:<6.2f} {r['threshold']:<6.2f} "
            f"{r['gap_penalty']:<6.1f} {r['chain_threshold']:<6.1f} "
            f"{r['min_detection_length']:<6} {r['precision']:<8.4f} "
            f"{r['recall']:<8.4f} {r['f1']:<8.4f}")

    logging.info(f"\nBEST: P={best_result['precision']:.4f} R={best_result['recall']:.4f} "
                 f"F1={best_result['f1']:.4f}")
    logging.info(f"Previous best: F1=0.5258")
    logging.info(f"Delta: {best_result['f1'] - 0.5258:+.4f}")

    # Save
    results_path = os.path.join(os.path.dirname(__file__), "exp5_fine_grid_results.json")
    with open(results_path, "w") as f:
        json.dump({"pair_limit": pair_limit, "total_configs": total_configs,
                    "best": best_result, "top10": top10,
                    "precompute_time": t_precompute, "sweep_time": t_sweep}, f, indent=2)
    logging.info(f"Saved to {results_path}")

    # Update trained_config if improved
    if best_result["f1"] > 0.5258:
        from config_utils import save_config
        save_config({
            "semantic_weight": best_result["semantic_weight"],
            "structural_weight": best_result["structural_weight"],
            "threshold": best_result["threshold"],
            "chain_threshold": best_result["chain_threshold"],
            "gap_penalty": best_result["gap_penalty"],
            "min_detection_length": best_result["min_detection_length"],
            "window_size": 150, "step_size": 25,
            "precision": best_result["precision"],
            "recall": best_result["recall"],
            "f1": best_result["f1"],
        })
        logging.info("Updated trained_config.json with new best!")

    append_experiment_log("Experiment 5: Fine Grid", {
        "dataset": dataset_path, "pair_limit": pair_limit,
        "configs_tested": total_configs,
        "best_f1": best_result["f1"], "best_config": best_result,
        "previous_best_f1": 0.5258,
    })


if __name__ == "__main__":
    main()
