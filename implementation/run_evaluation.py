import logging
import os
import sys
import multiprocessing as mp
from config_utils import load_config
from data_loader import PANDataLoader
from features import StructuralFeatureExtractor
from scoring import HybridScorer
from evaluator import PANEvaluator
from experiment_logger import append_experiment_log
from runtime_utils import load_sentence_transformer, resolve_dataset_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_DATASET = "pan25-generated-plagiarism-detection-validation.zip"


def _evaluate_single_pair(args):
    """Worker function for multiprocessing — processes one document pair."""
    susp_fn, src_fn, dataset_path, config, model_name, offline = args

    # Each worker loads its own model (avoids pickling issues)
    model = load_sentence_transformer(model_name, offline=offline)
    feature_extractor = StructuralFeatureExtractor()
    hybrid_scorer = HybridScorer(
        semantic_weight=config["semantic_weight"],
        structural_weight=config["structural_weight"],
    )
    evaluator = PANEvaluator(
        model, feature_extractor, hybrid_scorer,
        window_size=config["window_size"],
        step_size=config["step_size"],
    )
    loader = PANDataLoader(dataset_path)

    susp_text = loader.load_text(susp_fn, is_suspicious=True)
    src_text = loader.load_text(src_fn, is_suspicious=False)
    ground_truth = loader.load_truth(susp_fn, src_fn)

    detections = evaluator.detect_plagiarism(
        susp_text, src_text, threshold=config["threshold"],
    )
    metrics = evaluator.evaluate(detections, ground_truth, len(susp_text))

    return {
        'susp_fn': susp_fn,
        'src_fn': src_fn,
        'gt_count': len(ground_truth),
        'det_count': len(detections),
        **metrics,
    }


def run_evaluation(dataset_path=None, pair_limit=None, n_workers=None,
                   model_name="all-MiniLM-L6-v2", offline=False):
    logging.info("Initializing Evaluation Engine...")
    config = load_config()

    dataset_path = resolve_dataset_path(dataset_path, DEFAULT_DATASET)

    loader = PANDataLoader(dataset_path)
    pairs = loader.get_pairs()
    if pair_limit:
        pairs = pairs[:pair_limit]

    if not pairs:
        logging.error("No pairs found for evaluation.")
        return

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, len(pairs))

    logging.info(f"Evaluating {len(pairs)} pairs using {n_workers} workers...")

    # Build work items
    work_items = [
        (susp_fn, src_fn, dataset_path, config, model_name, offline)
        for susp_fn, src_fn in pairs
    ]

    all_results = []
    if n_workers <= 1:
        # Single-process fallback
        for item in work_items:
            result = _evaluate_single_pair(item)
            logging.info(
                f"  {result['susp_fn']} vs {result['src_fn']}: "
                f"P={result['precision']:.4f} R={result['recall']:.4f} F1={result['f1']:.4f}"
            )
            all_results.append(result)
    else:
        # Multi-process evaluation
        with mp.Pool(processes=n_workers) as pool:
            for result in pool.imap_unordered(_evaluate_single_pair, work_items):
                logging.info(
                    f"  {result['susp_fn']} vs {result['src_fn']}: "
                    f"P={result['precision']:.4f} R={result['recall']:.4f} F1={result['f1']:.4f}"
                )
                all_results.append(result)

    # Aggregate
    if all_results:
        avg_precision = sum(r['precision'] for r in all_results) / len(all_results)
        avg_recall = sum(r['recall'] for r in all_results) / len(all_results)
        avg_f1 = sum(r['f1'] for r in all_results) / len(all_results)
        logging.info("=" * 40)
        logging.info(f"AVERAGE MACRO PRECISION: {avg_precision:.4f}")
        logging.info(f"AVERAGE MACRO RECALL:    {avg_recall:.4f}")
        logging.info(f"AVERAGE MACRO F1 SCORE:  {avg_f1:.4f}")
        logging.info(f"Workers used: {n_workers}")
        logging.info("=" * 40)
        append_experiment_log(
            "Validation Run",
            {
                "dataset": dataset_path,
                "pair_limit": pair_limit if pair_limit is not None else "full",
                "workers": n_workers,
                "semantic_weight": config["semantic_weight"],
                "structural_weight": config["structural_weight"],
                "threshold": config["threshold"],
                "window_size": config["window_size"],
                "step_size": config["step_size"],
                "model_name": model_name,
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "avg_f1": avg_f1,
            },
        )


if __name__ == "__main__":
    dataset_arg = sys.argv[1] if len(sys.argv) > 1 else None
    pair_limit_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    workers_arg = int(sys.argv[3]) if len(sys.argv) > 3 else None
    model_arg = sys.argv[4] if len(sys.argv) > 4 else "all-MiniLM-L6-v2"
    offline_arg = (sys.argv[5].lower() == "true") if len(sys.argv) > 5 else False
    run_evaluation(dataset_arg, pair_limit_arg, workers_arg, model_arg, offline_arg)
