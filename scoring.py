import numpy as np

class HybridScorer:
    """
    Combines Semantic Similarity (SBERT) and Structural Similarity into a single Hybrid Score.
    This fulfills Task 2 of the thesis: 'propose a scientific article similarity approach 
    that combines semantic vector representations with syntactic alignment'.
    """
    def __init__(self, semantic_weight=0.7, structural_weight=0.3):
        """
        Initialize with weights for each component.
        Default: Semantic (SBERT) is weighted higher (0.7) than Structural (0.3).
        """
        self.w_sem = semantic_weight
        self.w_str = structural_weight
        
        # Verify weights sum to 1.0
        total = self.w_sem + self.w_str
        self.w_sem /= total
        self.w_str /= total

    def normalize_structural_diff(self, diff_dict):
        """
        Converts absolute differences into a 0-1 similarity score.
        A difference of 0 results in a similarity of 1.0.
        Large differences result in a similarity approaching 0.0.
        """
        # We use an exponential decay function: Sim = exp(-k * diff)
        # This is a common technique in IR and ML to map distances to similarities.
        
        # We weight specific structural features that are most indicative of similarity
        weights = {
            'word_count': 0.1,          # Scale down large count diffs
            'avg_word_length': 2.0,     # Very sensitive to small changes
            'avg_sentence_length': 0.5,
            'punctuation_ratio': 5.0,   # Ratios are small (0-1), so weight them high
            'stopword_ratio': 5.0,
            'noun_ratio': 5.0,
            'verb_ratio': 5.0,
            'adj_ratio': 5.0
        }
        
        weighted_diff_sum = 0
        for key, diff in diff_dict.items():
            weighted_diff_sum += diff * weights.get(key, 1.0)
            
        # Apply exponential decay
        # k=0.5 is a tuning parameter; lower k makes the scorer more 'forgiving'
        structural_similarity = np.exp(-0.5 * weighted_diff_sum)
        
        return structural_similarity

    def compute_hybrid_score(self, semantic_sim, structural_diffs):
        """
        Calculates the final Hybrid Similarity Score.
        """
        # 1. Normalize structural differences to a similarity score [0, 1]
        structural_sim = self.normalize_structural_diff(structural_diffs)
        
        # 2. Linear combination
        hybrid_score = (self.w_sem * semantic_sim) + (self.w_str * structural_sim)
        
        return {
            'hybrid_score': float(hybrid_score),
            'semantic_component': float(semantic_sim),
            'structural_component': float(structural_sim)
        }
