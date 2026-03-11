import logging
from sentence_transformers import SentenceTransformer
from data_loader import PANDataLoader
from features import StructuralFeatureExtractor
from scoring import HybridScorer
from evaluator import PANEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_evaluation():
    # 1. Initialize Components
    logging.info("Initializing Evaluation Engine...")
    
    # Model Loading (using MPS for Mac if available)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    feature_extractor = StructuralFeatureExtractor()
    hybrid_scorer = HybridScorer(semantic_weight=0.7, structural_weight=0.3)
    
    # Window settings: smaller step_size increases accuracy but decreases speed
    evaluator = PANEvaluator(model, feature_extractor, hybrid_scorer, window_size=150, step_size=25)
    
    # 2. Setup Data Path
    data_path = "/Users/shinthiya.promi/Desktop/MS_THESIS/implementation/data/pan2025_mock"
    loader = PANDataLoader(data_path)
    pairs = loader.get_pairs()

    if not pairs:
        logging.error("No pairs found for evaluation. Run mock_pan_generator.py first.")
        return

    logging.info(f"Found {len(pairs)} document pairs to evaluate.")

    all_results = []

    # 3. Process Each Pair
    for susp_fn, src_fn in pairs:
        logging.info("-" * 40)
        logging.info(f"Evaluating Pair: Suspicious({susp_fn}) vs Source({src_fn})")
        
        # Load content and truth
        susp_text = loader.load_text(susp_fn, is_suspicious=True)
        src_text = loader.load_text(src_fn, is_suspicious=False)
        ground_truth = loader.load_truth(susp_fn)

        # Detect
        detections = evaluator.detect_plagiarism(susp_text, src_text, threshold=0.45)
        
        # Evaluate
        metrics = evaluator.evaluate(detections, ground_truth, len(susp_text))
        
        logging.info(f"  > Detected {len(detections)} plagiarism cases.")
        logging.info(f"  > Metrics: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        
        all_results.append(metrics)

    # 4. Final Aggregation
    if all_results:
        avg_f1 = sum(r['f1'] for r in all_results) / len(all_results)
        logging.info("=" * 40)
        logging.info(f"AVERAGE MACRO F1 SCORE: {avg_f1:.4f}")
        logging.info("=" * 40)

if __name__ == "__main__":
    run_evaluation()
