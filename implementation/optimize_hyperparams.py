import logging
import os
import sys

import torch
from tqdm import tqdm

from config_utils import load_config, save_config
from data_loader import PANDataLoader
from evaluator import PANEvaluator
from features import StructuralFeatureExtractor
from runtime_utils import configure_cpu_runtime, load_sentence_transformer, resolve_dataset_path
from scoring import HybridScorer
from train_model import build_training_cache, evaluate_configuration, select_training_pairs


logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DATASET = os.path.join("extracted", "01_train", "01_train")
DEFAULT_PAIR_LIMIT = 500


def optimize_hyperparams(dataset_path=None, pair_limit=DEFAULT_PAIR_LIMIT, seed=42,
                         model_name="all-MiniLM-L6-v2", offline=False):
    if dataset_path is None:
        dataset_path = resolve_dataset_path(DEFAULT_DATASET, DEFAULT_DATASET)

    logging.info("Optimizing hyperparameters on balanced PAN train subset...")

    loader = PANDataLoader(dataset_path)
    pairs = select_training_pairs(loader, loader.get_pairs(), pair_limit, seed=seed)

    logging.info(
        "Using %d balanced training pairs from %s (seed=%d)",
        len(pairs),
        os.path.basename(os.path.normpath(dataset_path)),
        seed,
    )

    cpu_threads = os.cpu_count() or 1
    configure_cpu_runtime(max(1, min(4, cpu_threads)))
    logging.info("CPU threads configured: %d", cpu_threads)

    model = load_sentence_transformer(model_name, offline=offline)
    feature_extractor = StructuralFeatureExtractor()

    current_best = load_config()
    best_result = current_best.copy()
    best_f1 = current_best.get("f1", 0.0)

    window_candidates = [(120, 20), (150, 25), (180, 30)]
    semantic_candidates = [0.70, 0.80, 0.90]
    threshold_candidates = [0.50, 0.55, 0.60]

    results = []
    layout_progress = tqdm(window_candidates, desc="Layouts", unit="layout")

    for window_size, step_size in layout_progress:
        logging.info("-" * 78)
        logging.info("Evaluating layout window_size=%d step_size=%d", window_size, step_size)

        base_scorer = HybridScorer(semantic_weight=0.7, structural_weight=0.3)
        evaluator = PANEvaluator(
            model,
            feature_extractor,
            base_scorer,
            window_size=window_size,
            step_size=step_size,
        )
        cached_pairs = build_training_cache(loader, pairs, evaluator, model_name=model_name, offline=offline)

        config_progress = tqdm(
            total=len(semantic_candidates) * len(threshold_candidates),
            desc=f"Configs @ {window_size}/{step_size}",
            unit="config",
            leave=False,
        )

        for semantic_weight in semantic_candidates:
            structural_weight = 1.0 - semantic_weight
            for threshold in threshold_candidates:
                metrics = evaluate_configuration(
                    cached_pairs,
                    evaluator,
                    threshold,
                    semantic_weight,
                    structural_weight,
                )
                result = {
                    "semantic_weight": float(semantic_weight),
                    "structural_weight": float(structural_weight),
                    "threshold": float(threshold),
                    "window_size": int(window_size),
                    "step_size": int(step_size),
                    **metrics,
                }
                results.append(result)
                config_progress.update(1)
                logging.info(
                    "window=%-4d step=%-3d sem=%.2f str=%.2f thr=%.2f | P=%.4f R=%.4f F1=%.4f",
                    window_size,
                    step_size,
                    semantic_weight,
                    structural_weight,
                    threshold,
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                )

                if result["f1"] > best_f1:
                    best_result = result
                    best_f1 = result["f1"]
                    save_config(best_result)
                    logging.info(
                        "New best saved: sem=%.2f str=%.2f thr=%.2f window=%d step=%d F1=%.4f",
                        best_result["semantic_weight"],
                        best_result["structural_weight"],
                        best_result["threshold"],
                        best_result["window_size"],
                        best_result["step_size"],
                        best_result["f1"],
                    )

        config_progress.close()

    logging.info("=" * 78)
    logging.info(
        "Best hyperparameter result: sem=%.2f str=%.2f thr=%.2f window=%d step=%d F1=%.4f",
        best_result["semantic_weight"],
        best_result["structural_weight"],
        best_result["threshold"],
        best_result["window_size"],
        best_result["step_size"],
        best_result["f1"],
    )
    return best_result, results


if __name__ == "__main__":
    dataset_arg = sys.argv[1] if len(sys.argv) > 1 else None
    pair_limit_arg = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PAIR_LIMIT
    seed_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    model_arg = sys.argv[4] if len(sys.argv) > 4 else "all-MiniLM-L6-v2"
    offline_arg = (sys.argv[5].lower() == "true") if len(sys.argv) > 5 else False
    optimize_hyperparams(dataset_arg, pair_limit_arg, seed_arg, model_arg, offline_arg)
