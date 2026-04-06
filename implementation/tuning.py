import logging
import numpy as np
import os
import sys
from sentence_transformers import SentenceTransformer
from config_utils import load_config
from data_loader import PANDataLoader
from features import StructuralFeatureExtractor
from scoring import HybridScorer
from evaluator import PANEvaluator

logging.basicConfig(level=logging.INFO, format='%(message)s')

DEFAULT_DATASET = "pan25-generated-plagiarism-detection-train.zip"


def tune_threshold(dataset_path=None, pair_limit=None):
    logging.info("Starting Threshold Tuning...")
    config = load_config()
    
    # 1. Setup
    model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
    feature_extractor = StructuralFeatureExtractor()
    hybrid_scorer = HybridScorer(
        semantic_weight=config["semantic_weight"],
        structural_weight=config["structural_weight"],
    )
    evaluator = PANEvaluator(
        model,
        feature_extractor,
        hybrid_scorer,
        window_size=config["window_size"],
        step_size=config["step_size"],
    )
    
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), "data", DEFAULT_DATASET)

    loader = PANDataLoader(dataset_path)
    pairs = loader.get_pairs()
    if pair_limit:
        pairs = pairs[:pair_limit]
    
    if not pairs:
        logging.error("No data found.")
        return

    # 2. Iterate through thresholds
    thresholds = np.arange(0.1, 0.6, 0.05)
    best_f1 = 0
    best_threshold = 0
    
    logging.info(f"{'Threshold':<12} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    logging.info("-" * 50)

    for threshold in thresholds:
        all_metrics = []
        for susp_fn, src_fn in pairs:
            susp_text = loader.load_text(susp_fn, is_suspicious=True)
            src_text = loader.load_text(src_fn, is_suspicious=False)
            ground_truth = loader.load_truth(susp_fn, src_fn)
            
            detections = evaluator.detect_plagiarism(susp_text, src_text, threshold=threshold)
            metrics = evaluator.evaluate(detections, ground_truth, len(susp_text))
            all_metrics.append(metrics)
        
        avg_p = sum(m['precision'] for m in all_metrics) / len(all_metrics)
        avg_r = sum(m['recall'] for m in all_metrics) / len(all_metrics)
        avg_f1 = sum(m['f1'] for m in all_metrics) / len(all_metrics)
        
        logging.info(f"{threshold:<12.2f} | {avg_p:<10.4f} | {avg_r:<10.4f} | {avg_f1:<10.4f}")
        
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_threshold = threshold

    logging.info("-" * 50)
    logging.info(f"OPTIMAL THRESHOLD: {best_threshold:.2f} (F1: {best_f1:.4f})")

if __name__ == "__main__":
    dataset_arg = sys.argv[1] if len(sys.argv) > 1 else None
    pair_limit_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    tune_threshold(dataset_arg, pair_limit_arg)
