"""Quick mpnet vs MiniLM comparison using best Phase 2 config, no multiprocessing."""
import json
import time
import logging
import argparse

import numpy as np
from tqdm import tqdm

from data_loader import PANDataLoader
from evaluator import PANEvaluator
from features import StructuralFeatureExtractor
from runtime_utils import load_sentence_transformer, resolve_dataset_path
from scoring import HybridScorer
from experiment_logger import append_experiment_log

logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DATASET = "pan25-generated-plagiarism-detection-validation.zip"


def evaluate_model(model_name, pairs, config, dataset_path, offline=False):
    logging.info(f"\n{'='*60}")
    logging.info(f"Testing model: {model_name}")
    logging.info(f"{'='*60}")

    model = load_sentence_transformer(model_name, offline=offline)
    fe = StructuralFeatureExtractor()
    hs = HybridScorer(semantic_weight=config["semantic_weight"],
                      structural_weight=config["structural_weight"])
    ev = PANEvaluator(model, fe, hs, window_size=config["window_size"],
                      step_size=config["step_size"])
    ld = PANDataLoader(dataset_path)

    precisions, recalls, f1s = [], [], []
    t0 = time.time()

    for susp_fn, src_fn in tqdm(pairs, desc=f"Evaluating ({model_name})", unit="pair"):
        susp_text = ld.load_text(susp_fn, is_suspicious=True)
        src_text = ld.load_text(src_fn, is_suspicious=False)
        ground_truth = ld.load_truth(susp_fn, src_fn)
        pair_data = ev.precompute_pair_data(susp_text, src_text)

        dets = ev.detect_plagiarism_from_precomputed(
            pair_data,
            threshold=config["threshold"],
            gap_penalty=config["gap_penalty"],
            semantic_weight=config["semantic_weight"],
            structural_weight=config["structural_weight"],
            chain_threshold=config["chain_threshold"],
            min_detection_length=config["min_detection_length"],
        )
        metrics = ev.evaluate(dets, ground_truth, len(susp_text))
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1s.append(metrics["f1"])

    elapsed = time.time() - t0
    n = len(pairs)
    result = {
        "model": model_name,
        "precision": sum(precisions) / n,
        "recall": sum(recalls) / n,
        "f1": sum(f1s) / n,
        "precompute_time": elapsed,
    }
    logging.info(f"  Precision: {result['precision']:.4f}")
    logging.info(f"  Recall:    {result['recall']:.4f}")
    logging.info(f"  F1:        {result['f1']:.4f}")
    logging.info(f"  Time:      {elapsed:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pair_limit", nargs="?", type=int, default=50)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    dataset_path = resolve_dataset_path(args.dataset, DEFAULT_DATASET)

    with open("trained_config.json") as f:
        config = json.load(f)

    logging.info("Config: %s", json.dumps(config, indent=2))

    loader = PANDataLoader(dataset_path)
    pairs = loader.get_pairs()[:args.pair_limit]
    logging.info(f"Pairs: {len(pairs)}")

    minilm = evaluate_model("all-MiniLM-L6-v2", pairs, config, dataset_path, args.offline)
    mpnet = evaluate_model("all-mpnet-base-v2", pairs, config, dataset_path, args.offline)

    winner = "all-MiniLM-L6-v2" if minilm["f1"] >= mpnet["f1"] else "all-mpnet-base-v2"

    logging.info(f"\n{'='*60}")
    logging.info(f"WINNER: {winner}")
    logging.info(f"  MiniLM F1: {minilm['f1']:.4f}")
    logging.info(f"  mpnet  F1: {mpnet['f1']:.4f}")
    logging.info(f"{'='*60}")

    comparison = {"minilm": minilm, "mpnet": mpnet, "winner": winner}
    with open("model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    append_experiment_log("Model Comparison", {
        "dataset": dataset_path,
        "pair_limit": args.pair_limit,
        "config": config,
        "minilm_f1": minilm["f1"],
        "mpnet_f1": mpnet["f1"],
        "winner": winner,
    })


if __name__ == "__main__":
    main()
