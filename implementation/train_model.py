import logging
import os
import sys
import json

import numpy as np
import torch
from tqdm import tqdm

from config_utils import save_config
from data_loader import PANDataLoader
from evaluator import PANEvaluator
from experiment_logger import append_experiment_log
from features import StructuralFeatureExtractor
from runtime_utils import configure_cpu_runtime, load_sentence_transformer, resolve_dataset_path
from scoring import HybridScorer


logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DATASET = "pan25-generated-plagiarism-detection-train.zip"
DEFAULT_SEED = 42
STREAMING_CHECKPOINT_INTERVAL = 100


def get_label_cache_path(dataset_path):
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    return os.path.join(
        os.path.dirname(__file__),
        f".pair_labels_{dataset_name}.json",
    )


def sample_pairs(pairs, pair_limit, seed=DEFAULT_SEED, oversample_factor=500):
    if not pair_limit or pair_limit >= len(pairs):
        return pairs

    rng = np.random.default_rng(seed)
    pool_size = min(len(pairs), max(pair_limit, pair_limit * oversample_factor))
    indices = rng.choice(len(pairs), size=pool_size, replace=False)
    indices.sort()
    return [pairs[idx] for idx in indices]


def prioritize_smaller_pairs(loader, pairs, pair_limit):
    if not pair_limit or pair_limit >= len(pairs):
        return pairs

    ranked = sorted(
        pairs,
        key=lambda pair: (
            loader.get_text_size(pair[0], True) + loader.get_text_size(pair[1], False),
            pair[0],
            pair[1],
        ),
    )
    return ranked[:pair_limit]


def stratify_pairs_by_truth(loader, pairs):
    cache_path = get_label_cache_path(loader.base_path)
    label_cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as handle:
            label_cache = json.load(handle)

    positive_pairs = []
    negative_pairs = []
    cache_updated = False

    for susp_fn, src_fn in tqdm(pairs, desc="Scanning pair labels", unit="pair", leave=False):
        pair_key = f"{susp_fn}|{src_fn}"
        has_truth = label_cache.get(pair_key)
        if has_truth is None:
            has_truth = bool(loader.load_truth(susp_fn, src_fn))
            label_cache[pair_key] = has_truth
            cache_updated = True

        if has_truth:
            positive_pairs.append((susp_fn, src_fn))
        else:
            negative_pairs.append((susp_fn, src_fn))

    if cache_updated:
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(label_cache, handle)

    return positive_pairs, negative_pairs


def select_training_pairs(loader, all_pairs, pair_limit, seed=DEFAULT_SEED):
    if not pair_limit or pair_limit >= len(all_pairs):
        return all_pairs

    positive_pairs, negative_pairs = stratify_pairs_by_truth(loader, all_pairs)
    target_positive = min(len(positive_pairs), max(1, pair_limit // 2))
    target_negative = min(len(negative_pairs), pair_limit - target_positive)

    rng = np.random.default_rng(seed)

    def sample_group(group, limit):
        if limit <= 0 or not group:
            return []
        if len(group) <= limit:
            return prioritize_smaller_pairs(loader, group, limit)
        indices = rng.choice(len(group), size=min(len(group), limit * 200), replace=False)
        pool = [group[idx] for idx in sorted(indices)]
        return prioritize_smaller_pairs(loader, pool, limit)

    selected_positive = sample_group(positive_pairs, target_positive)
    selected_negative = sample_group(negative_pairs, target_negative)
    selected_pairs = selected_positive + selected_negative

    if len(selected_pairs) < pair_limit:
        remaining = [pair for pair in all_pairs if pair not in selected_pairs]
        selected_pairs.extend(prioritize_smaller_pairs(loader, remaining, pair_limit - len(selected_pairs)))

    return selected_pairs


def _precompute_one_pair(args):
    """Worker function: precompute embeddings + features for one pair."""
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


def build_training_cache(loader, pairs, evaluator, n_workers=None,
                         model_name="all-MiniLM-L6-v2", offline=False):
    import multiprocessing as mp

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, len(pairs))

    if n_workers <= 1:
        # Single-process fallback
        cached_pairs = []
        for susp_fn, src_fn in tqdm(pairs, desc="Precomputing pairs", unit="pair"):
            susp_text = loader.load_text(susp_fn, is_suspicious=True)
            src_text = loader.load_text(src_fn, is_suspicious=False)
            ground_truth = loader.load_truth(susp_fn, src_fn)
            pair_data = evaluator.precompute_pair_data(susp_text, src_text)
            cached_pairs.append({
                "susp_fn": susp_fn,
                "src_fn": src_fn,
                "ground_truth": ground_truth,
                "pair_data": pair_data,
                "text_length": len(susp_text),
            })
        return cached_pairs

    logging.info(f"Precomputing pairs using {n_workers} workers...")
    work_items = [
        (susp_fn, src_fn, loader.base_path,
         evaluator.window_size, evaluator.step_size, model_name, offline)
        for susp_fn, src_fn in pairs
    ]

    cached_pairs = []
    with mp.Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap(_precompute_one_pair, work_items),
            total=len(work_items),
            desc="Precomputing pairs (parallel)",
            unit="pair",
        ):
            cached_pairs.append(result)

    return cached_pairs


def evaluate_configuration(cached_pairs, evaluator, threshold, semantic_weight, structural_weight):
    # Vectorised evaluator makes this fast enough single-threaded
    precisions = []
    recalls = []
    f1s = []

    for cached in cached_pairs:
        detections = evaluator.detect_plagiarism_from_precomputed(
            cached["pair_data"],
            threshold=threshold,
            semantic_weight=semantic_weight,
            structural_weight=structural_weight,
        )
        metrics = evaluator.evaluate(detections, cached["ground_truth"], cached["text_length"])
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1s.append(metrics["f1"])

    n = len(cached_pairs)
    return {
        "precision": sum(precisions) / n,
        "recall": sum(recalls) / n,
        "f1": sum(f1s) / n,
    }


def initialize_config_results(semantic_weights, thresholds):
    config_results = []
    for semantic_weight in semantic_weights:
        structural_weight = 1.0 - semantic_weight
        for threshold in thresholds:
            config_results.append({
                "semantic_weight": float(semantic_weight),
                "structural_weight": float(structural_weight),
                "threshold": float(threshold),
                "window_size": 150,
                "step_size": 25,
                "precision_sum": 0.0,
                "recall_sum": 0.0,
                "f1_sum": 0.0,
                "count": 0,
            })
    return config_results


def finalize_result(config_result):
    count = max(1, config_result["count"])
    return {
        "semantic_weight": config_result["semantic_weight"],
        "structural_weight": config_result["structural_weight"],
        "threshold": config_result["threshold"],
        "window_size": config_result["window_size"],
        "step_size": config_result["step_size"],
        "precision": config_result["precision_sum"] / count,
        "recall": config_result["recall_sum"] / count,
        "f1": config_result["f1_sum"] / count,
    }


def train_model_streaming(loader, pairs, evaluator, semantic_weights, thresholds):
    config_results = initialize_config_results(semantic_weights, thresholds)
    pair_progress = tqdm(pairs, desc="Streaming full train", unit="pair")

    for pair_index, (susp_fn, src_fn) in enumerate(pair_progress, start=1):
        susp_text = loader.load_text(susp_fn, is_suspicious=True)
        src_text = loader.load_text(src_fn, is_suspicious=False)
        ground_truth = loader.load_truth(susp_fn, src_fn)
        pair_data = evaluator.precompute_pair_data(susp_text, src_text)

        for config_result in config_results:
            detections = evaluator.detect_plagiarism_from_precomputed(
                pair_data,
                threshold=config_result["threshold"],
                semantic_weight=config_result["semantic_weight"],
                structural_weight=config_result["structural_weight"],
            )
            metrics = evaluator.evaluate(detections, ground_truth, len(susp_text))
            config_result["precision_sum"] += metrics["precision"]
            config_result["recall_sum"] += metrics["recall"]
            config_result["f1_sum"] += metrics["f1"]
            config_result["count"] += 1

        best_so_far = max((finalize_result(result) for result in config_results), key=lambda item: item["f1"])
        pair_progress.set_postfix(
            best_f1=f"{best_so_far['f1']:.4f}",
            sem=f"{best_so_far['semantic_weight']:.2f}",
            thr=f"{best_so_far['threshold']:.2f}",
        )

        if pair_index % STREAMING_CHECKPOINT_INTERVAL == 0:
            save_config(best_so_far)
            logging.info(
                "Checkpoint after %d pairs: best semantic_weight=%.2f structural_weight=%.2f threshold=%.2f F1=%.4f",
                pair_index,
                best_so_far["semantic_weight"],
                best_so_far["structural_weight"],
                best_so_far["threshold"],
                best_so_far["f1"],
            )

    return [finalize_result(result) for result in config_results]


def train_model(dataset_path=None, pair_limit=None, seed=DEFAULT_SEED):
    logging.info("Training hybrid parameters on PAN data...")

    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), "data", DEFAULT_DATASET)

    loader = PANDataLoader(dataset_path)
    pairs = select_training_pairs(loader, loader.get_pairs(), pair_limit, seed=seed)

    if not pairs:
        logging.error("No training pairs found.")
        return None

    logging.info(
        f"Using {len(pairs)} training pairs from {os.path.basename(dataset_path)} "
        f"(seed={seed})"
    )

    cpu_threads = os.cpu_count() or 1
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(max(1, min(4, cpu_threads)))
    logging.info(f"CPU threads configured: {cpu_threads}")

    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    feature_extractor = StructuralFeatureExtractor()
    base_scorer = HybridScorer(semantic_weight=0.7, structural_weight=0.3)
    evaluator = PANEvaluator(
        model,
        feature_extractor,
        base_scorer,
        window_size=150,
        step_size=25,
    )

    semantic_weights = [0.6, 0.7, 0.8]
    thresholds = [0.35, 0.45, 0.55]
    logging.info(f"{'SemW':<8} | {'StrW':<8} | {'Thresh':<8} | {'Prec':<8} | {'Recall':<8} | {'F1':<8}")
    logging.info("-" * 68)
    if pair_limit is None:
        all_results = train_model_streaming(
            loader,
            pairs,
            evaluator,
            semantic_weights,
            thresholds,
        )
    else:
        cached_pairs = build_training_cache(loader, pairs, evaluator)
        all_results = []
        total_configs = len(semantic_weights) * len(thresholds)
        config_progress = tqdm(total=total_configs, desc="Configs", unit="config")

        for semantic_weight in semantic_weights:
            structural_weight = 1.0 - semantic_weight
            for threshold in thresholds:
                metrics = evaluate_configuration(
                    cached_pairs,
                    evaluator,
                    threshold,
                    semantic_weight,
                    structural_weight,
                )
                logging.info(
                    f"{semantic_weight:<8.2f} | {structural_weight:<8.2f} | {threshold:<8.2f} | "
                    f"{metrics['precision']:<8.4f} | {metrics['recall']:<8.4f} | {metrics['f1']:<8.4f}"
                )
                config_progress.update(1)
                all_results.append({
                    "semantic_weight": float(semantic_weight),
                    "structural_weight": float(structural_weight),
                    "threshold": float(threshold),
                    "window_size": 150,
                    "step_size": 25,
                    **metrics,
                })

        config_progress.close()

    best_result = max(all_results, key=lambda item: item["f1"])

    for result in all_results:
        logging.info(
            f"{result['semantic_weight']:<8.2f} | {result['structural_weight']:<8.2f} | "
            f"{result['threshold']:<8.2f} | {result['precision']:<8.4f} | "
            f"{result['recall']:<8.4f} | {result['f1']:<8.4f}"
        )

    logging.info("-" * 68)
    logging.info(
        "Best config: semantic_weight=%.2f structural_weight=%.2f threshold=%.2f F1=%.4f",
        best_result["semantic_weight"],
        best_result["structural_weight"],
        best_result["threshold"],
        best_result["f1"],
    )

    path = save_config(best_result)
    logging.info("Saved trained config to %s", path)
    append_experiment_log(
        "Training Run",
        {
            "dataset": dataset_path,
            "pair_limit": pair_limit if pair_limit is not None else "full",
            "semantic_weight": best_result["semantic_weight"],
            "structural_weight": best_result["structural_weight"],
            "threshold": best_result["threshold"],
            "window_size": best_result["window_size"],
            "step_size": best_result["step_size"],
            "precision": best_result["precision"],
            "recall": best_result["recall"],
            "f1": best_result["f1"],
        },
    )
    return best_result


if __name__ == "__main__":
    dataset_arg = sys.argv[1] if len(sys.argv) > 1 else None
    pair_limit_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    seed_arg = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_SEED
    train_model(dataset_arg, pair_limit_arg, seed_arg)
