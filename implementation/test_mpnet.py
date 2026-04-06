"""
Test all-mpnet-base-v2 vs all-MiniLM-L6-v2 with the best Phase 2 config.
Uses multiprocessing for pair precomputation.
"""

import os
import time
import json
import logging
import argparse
import multiprocessing as mp

from tqdm import tqdm

from data_loader import PANDataLoader
from evaluator import PANEvaluator
from experiment_logger import append_experiment_log
from features import StructuralFeatureExtractor
from runtime_utils import (
    configure_cpu_runtime,
    load_sentence_transformer,
    resolve_dataset_path,
    resolve_device,
)
from scoring import HybridScorer

logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DATASET = "pan25-generated-plagiarism-detection-validation.zip"


def _configure_worker_runtime(device):
    # Avoid CPU oversubscription: multiprocessing is already providing parallelism.
    if device == "cpu":
        configure_cpu_runtime(1)


def _precompute_worker(args):
    susp_fn, src_fn, dataset_path, model_name, window_size, step_size, device, offline = args
    _configure_worker_runtime(device)
    model = load_sentence_transformer(model_name, device=device, offline=offline)
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


def evaluate_model(model_name, pairs, config, n_workers, device, dataset_path, offline=False):
    logging.info(f"\n{'='*60}")
    logging.info(f"Testing model: {model_name}")
    logging.info(f"{'='*60}")

    work_items = [
        (susp_fn, src_fn, dataset_path, model_name, 150, 25, device, offline)
        for susp_fn, src_fn in pairs
    ]

    t0 = time.time()
    cached_pairs = []
    with mp.Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap(_precompute_worker, work_items),
            total=len(work_items),
            desc=f"Precomputing ({model_name})",
            unit="pair",
        ):
            cached_pairs.append(result)
    t_precompute = time.time() - t0

    # Evaluate with best Phase 2 config
    _configure_worker_runtime(device)
    model = load_sentence_transformer(model_name, device=device, offline=offline)
    fe = StructuralFeatureExtractor()
    hs = HybridScorer(
        semantic_weight=config["semantic_weight"],
        structural_weight=config["structural_weight"],
    )
    evaluator = PANEvaluator(model, fe, hs, window_size=150, step_size=25)

    precisions, recalls, f1s = [], [], []
    for cached in cached_pairs:
        dets = evaluator.detect_plagiarism_from_precomputed(
            cached["pair_data"],
            threshold=config["threshold"],
            semantic_weight=config["semantic_weight"],
            structural_weight=config["structural_weight"],
            gap_penalty=config.get("gap_penalty", -0.5),
            chain_threshold=config.get("chain_threshold", 0.1),
            min_detection_length=config.get("min_detection_length", 0),
        )
        m = evaluator.evaluate(dets, cached["ground_truth"], cached["text_length"])
        precisions.append(m["precision"])
        recalls.append(m["recall"])
        f1s.append(m["f1"])

    n = len(cached_pairs)
    result = {
        "model": model_name,
        "precision": sum(precisions) / n,
        "recall": sum(recalls) / n,
        "f1": sum(f1s) / n,
        "precompute_time": t_precompute,
    }

    logging.info(f"  Precision: {result['precision']:.4f}")
    logging.info(f"  Recall:    {result['recall']:.4f}")
    logging.info(f"  F1:        {result['f1']:.4f}")
    logging.info(f"  Time:      {t_precompute:.1f}s")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pair_limit", nargs="?", type=int, default=50)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--only-model",
        choices=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
        default=None,
    )
    args = parser.parse_args()

    pair_limit = args.pair_limit
    n_workers = args.workers
    device = resolve_device(args.device)
    dataset_path = resolve_dataset_path(args.dataset, DEFAULT_DATASET)

    # Load best Phase 2 config
    with open(os.path.join(os.path.dirname(__file__), "trained_config.json")) as f:
        config = json.load(f)

    logging.info(f"Best Phase 2 config: sem={config['semantic_weight']}, "
                 f"thr={config['threshold']}, gap={config.get('gap_penalty', -0.5)}, "
                 f"chain={config.get('chain_threshold', 0.1)}, "
                 f"minlen={config.get('min_detection_length', 0)}")
    logging.info(f"Device: {device}")
    if device == "cpu":
        logging.info("CPU mode: setting PyTorch intra/inter-op threads to 1 per worker")

    loader = PANDataLoader(dataset_path)
    pairs = loader.get_pairs()[:pair_limit]
    logging.info(f"Pairs: {len(pairs)} | Workers: {n_workers}")

    minilm = None
    mpnet = None
    if args.only_model in (None, "all-MiniLM-L6-v2"):
        minilm = evaluate_model("all-MiniLM-L6-v2", pairs, config, n_workers, device, dataset_path, args.offline)
    if args.only_model in (None, "all-mpnet-base-v2"):
        mpnet = evaluate_model("all-mpnet-base-v2", pairs, config, n_workers, device, dataset_path, args.offline)

    # Summary
    logging.info(f"\n{'='*60}")
    logging.info("MODEL COMPARISON")
    logging.info(f"{'='*60}")
    logging.info(f"{'Model':<25} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Time':<8}")
    logging.info("-" * 60)
    if minilm:
        logging.info(f"{'all-MiniLM-L6-v2':<25} {minilm['precision']:<8.4f} "
                     f"{minilm['recall']:<8.4f} {minilm['f1']:<8.4f} {minilm['precompute_time']:<8.1f}s")
    if mpnet:
        logging.info(f"{'all-mpnet-base-v2':<25} {mpnet['precision']:<8.4f} "
                     f"{mpnet['recall']:<8.4f} {mpnet['f1']:<8.4f} {mpnet['precompute_time']:<8.1f}s")

    if minilm and mpnet:
        winner = mpnet if mpnet["f1"] > minilm["f1"] else minilm
        logging.info(f"\nWinner: {winner['model']} (F1={winner['f1']:.4f})")

        append_experiment_log(
            "Model Comparison",
            {
                "dataset": dataset_path,
                "pair_limit": pair_limit,
                "device": device,
                "workers": n_workers,
                "config": config,
                "minilm_f1": minilm["f1"],
                "mpnet_f1": mpnet["f1"],
                "winner": winner["model"],
            },
        )

        comp_path = os.path.join(os.path.dirname(__file__), "model_comparison.json")
        with open(comp_path, "w") as f:
            json.dump({"minilm": minilm, "mpnet": mpnet, "winner": winner["model"]}, f, indent=2)
        logging.info(f"Saved to {comp_path}")


if __name__ == "__main__":
    main()
