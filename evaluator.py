import logging
import numpy as np
from tqdm import tqdm
from sentence_transformers import util

class PANEvaluator:
    """
    Evaluates the Hybrid Similarity Model on PAN-style datasets.
    Implements character-level Precision, Recall, and F1 as per PAN standards.
    """
    def __init__(self, model, feature_extractor, hybrid_scorer, window_size=250, step_size=125):
        """
        window_size: Character length of the sliding window.
        step_size: Number of characters to shift the window.
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.hybrid_scorer = hybrid_scorer
        self.window_size = window_size
        self.step_size = step_size

    def get_windows(self, text):
        """Generates sliding windows from text with their offsets."""
        windows = []
        for i in range(0, max(1, len(text) - self.window_size + 1), self.step_size):
            window_text = text[i : i + self.window_size]
            windows.append({'text': window_text, 'offset': i, 'length': len(window_text)})
        return windows

    def detect_plagiarism(self, susp_text, src_text, threshold=0.4, gap_penalty=-0.5):
        """
        Scans suspicious text against source text using Semantic Smith-Waterman alignment.
        This captures global sequences and handles scrambled text.
        """
        susp_windows = self.get_windows(susp_text)
        src_windows = self.get_windows(src_text)

        if not susp_windows or not src_windows:
            return []

        # 1. Precompute all embeddings and features
        src_texts = [w['text'] for w in src_windows]
        src_embeddings = self.model.encode(src_texts, convert_to_tensor=True, show_progress_bar=False)
        src_features = [self.feature_extractor.extract_features(w['text']) for w in src_windows]

        susp_texts = [w['text'] for w in susp_windows]
        susp_embeddings = self.model.encode(susp_texts, convert_to_tensor=True, show_progress_bar=False)
        susp_features = [self.feature_extractor.extract_features(w['text']) for w in susp_windows]

        # 2. Compute Hybrid Similarity Matrix (M x N)
        # Using SBERT's batch cosine similarity
        cos_sims = util.pytorch_cos_sim(susp_embeddings, src_embeddings).cpu().numpy()
        
        m, n = len(susp_windows), len(src_windows)
        score_matrix = np.zeros((m + 1, n + 1))
        
        # We find the best path using Smith-Waterman
        for i in range(1, m + 1):
            s_feat = susp_features[i-1]
            for j in range(1, n + 1):
                # Calculate Hybrid Score for this pair
                str_diff = self.feature_extractor.get_feature_diff(s_feat, src_features[j-1])
                hybrid_res = self.hybrid_scorer.compute_hybrid_score(cos_sims[i-1][j-1], str_diff)
                match_score = hybrid_res['hybrid_score']
                
                # Smith-Waterman logic: Max of 0, Match, or Gaps
                score_matrix[i, j] = max(
                    0,
                    score_matrix[i-1, j-1] + (match_score if match_score >= threshold else -1.0),
                    score_matrix[i-1, j] + gap_penalty,
                    score_matrix[i, j-1] + gap_penalty
                )

        # 3. Traceback to find aligned segments
        detections = []
        # Find all local maxima above a certain 'chain' threshold
        chain_threshold = 0.1 # Lowered for very small mock data
        
        # Simple extraction of high-scoring segments from the matrix
        # For a full SOTA implementation, we would do a proper traceback.
        # For this prototype, we identify windows that contributed to a high score chain.
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if score_matrix[i, j] > chain_threshold:
                    # If this cell is part of a local peak
                    if score_matrix[i, j] >= score_matrix[i-1, j] and score_matrix[i, j] >= score_matrix[i, j-1]:
                        detections.append({
                            'this_offset': susp_windows[i-1]['offset'],
                            'this_length': susp_windows[i-1]['length'],
                            'source_offset': src_windows[j-1]['offset'],
                            'source_length': src_windows[j-1]['length'],
                            'score': float(score_matrix[i, j])
                        })

        return self.merge_detections(detections)

    def merge_detections(self, detections):
        """Simplistic merging of adjacent detected windows."""
        if not detections: return []
        
        merged = []
        current = detections[0].copy()
        
        for next_det in detections[1:]:
            # If windows are adjacent or overlap
            if next_det['this_offset'] <= current['this_offset'] + current['this_length']:
                # Update length to cover both
                current['this_length'] = (next_det['this_offset'] + next_det['this_length']) - current['this_offset']
                # (Simplification: we don't update source offsets for this prototype)
            else:
                merged.append(current)
                current = next_det.copy()
        
        merged.append(current)
        return merged

    def evaluate(self, detections, truths, text_length):
        """
        Calculates Precision, Recall, and F1 at the character level.
        """
        # Create binary arrays for predicted and ground truth plagiarism
        pred_arr = np.zeros(text_length, dtype=int)
        true_arr = np.zeros(text_length, dtype=int)

        for d in detections:
            pred_arr[d['this_offset'] : d['this_offset'] + d['this_length']] = 1
        
        for t in truths:
            true_arr[t['this_offset'] : t['this_offset'] + t['this_length']] = 1

        # Character-level stats
        tp = np.sum((pred_arr == 1) & (true_arr == 1))
        fp = np.sum((pred_arr == 1) & (true_arr == 0))
        fn = np.sum((pred_arr == 0) & (true_arr == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
