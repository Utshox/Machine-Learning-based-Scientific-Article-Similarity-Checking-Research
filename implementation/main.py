import logging
from sentence_transformers import SentenceTransformer, util
from features import StructuralFeatureExtractor
from scoring import HybridScorer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Final Integrated Prototype: Semantic + Structural = Hybrid Score.
    """
    logging.info("Initializing Thesis Hybrid Model...")
    
    # Load SBERT model
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    # Initialize Feature Extractor and Hybrid Scorer
    feature_extractor = StructuralFeatureExtractor()
    # 70% Semantic, 30% Structural as per common hybrid model research
    hybrid_scorer = HybridScorer(semantic_weight=0.7, structural_weight=0.3)

    # ---------------------------------------------------------
    # Test Scenarios
    # ---------------------------------------------------------
    
    # 1. Original vs Paraphrased (High Similarity)
    original = "Machine learning algorithms can identify patterns in large datasets."
    paraphrased = "Deep learning techniques are effective at finding structures in big data."

    # 2. Original vs Unrelated (Low Similarity)
    unrelated = "This study analyzes the impact of climate change on biodiversity in the Amazon rainforest."

    logging.info("-" * 60)
    logging.info(f"Original:  {original}")
    logging.info(f"Paraphrase: {paraphrased}")
    logging.info(f"Unrelated:  {unrelated}")
    logging.info("-" * 60)

    # ---------------------------------------------------------
    # Analysis Workflow
    # ---------------------------------------------------------
    
    # Embeddings
    embeddings = model.encode([original, paraphrased, unrelated], convert_to_tensor=True)
    
    # Features
    feats_orig = feature_extractor.extract_features(original)
    feats_para = feature_extractor.extract_features(paraphrased)
    feats_unrel = feature_extractor.extract_features(unrelated)

    # 1. Compare Original vs Paraphrased
    sem_sim_12 = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    diff_12 = feature_extractor.get_feature_diff(feats_orig, feats_para)
    results_12 = hybrid_scorer.compute_hybrid_score(sem_sim_12, diff_12)

    # 2. Compare Original vs Unrelated
    sem_sim_13 = util.pytorch_cos_sim(embeddings[0], embeddings[2]).item()
    diff_13 = feature_extractor.get_feature_diff(feats_orig, feats_unrel)
    results_13 = hybrid_scorer.compute_hybrid_score(sem_sim_13, diff_13)

    # ---------------------------------------------------------
    # Output Results
    # ---------------------------------------------------------
    
    logging.info("HYBRID SCORING RESULTS (Semantic + Structural):")
    logging.info("-" * 60)
    
    logging.info("SCENARIO: Original vs Paraphrased (High Similarity Expectation)")
    logging.info(f"  > Semantic (SBERT) Score: {results_12['semantic_component']:.4f}")
    logging.info(f"  > Structural (Normalized): {results_12['structural_component']:.4f}")
    logging.info(f"  ==> FINAL HYBRID SCORE:  {results_12['hybrid_score']:.4f}")
    
    logging.info("-" * 30)
    
    logging.info("SCENARIO: Original vs Unrelated (Low Similarity Expectation)")
    logging.info(f"  > Semantic (SBERT) Score: {results_13['semantic_component']:.4f}")
    logging.info(f"  > Structural (Normalized): {results_13['structural_component']:.4f}")
    logging.info(f"  ==> FINAL HYBRID SCORE:  {results_13['hybrid_score']:.4f}")
    
    logging.info("-" * 60)
    logging.info("Implementation Task 3: Core Hybrid Logic Completed Successfully.")

if __name__ == "__main__":
    main()
