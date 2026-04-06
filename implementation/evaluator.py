import logging
import numpy as np
from sentence_transformers import util

class PANEvaluator:
    """
    Evaluates the Hybrid Similarity Model on PAN-style datasets.
    Implements character-level Precision, Recall, and F1 as per PAN standards.
    Optimised for multi-core Apple Silicon (M4) via NumPy vectorisation.
    """
    def __init__(self, model, feature_extractor, hybrid_scorer, window_size=250, step_size=125):
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

    def _extract_features_batch(self, windows):
        """Extract features for all windows and return as a 2D NumPy array."""
        feature_keys = [
            'word_count', 'avg_word_length', 'avg_sentence_length',
            'punctuation_ratio', 'stopword_ratio',
            'noun_ratio', 'verb_ratio', 'adj_ratio'
        ]
        features_list = [self.feature_extractor.extract_features(w["text"]) for w in windows]
        features_array = np.array([[f[k] for k in feature_keys] for f in features_list])
        return features_array, feature_keys

    def _compute_structural_matrix_vectorised(self, susp_features, src_features):
        """
        Compute the full m x n structural similarity matrix using vectorised NumPy.
        Replaces the O(m*n) Python loop with broadcasting.
        """
        # Feature-specific weights matching scoring.py
        weight_values = np.array([0.1, 2.0, 0.5, 5.0, 5.0, 5.0, 5.0, 5.0])

        # susp_features: (m, 8), src_features: (n, 8)
        # diff: (m, n, 8) via broadcasting
        diff = np.abs(susp_features[:, np.newaxis, :] - src_features[np.newaxis, :, :])

        # Weighted sum across features: (m, n)
        weighted_diff = np.sum(diff * weight_values[np.newaxis, np.newaxis, :], axis=2)

        # Exponential decay: same as scoring.py normalize_structural_diff
        structural_matrix = np.exp(-0.5 * weighted_diff)

        return structural_matrix

    def precompute_pair_data(self, susp_text, src_text):
        """
        Precomputes windowing, embeddings, and pairwise similarity matrices once.
        Uses vectorised structural matrix computation.
        """
        susp_windows = self.get_windows(susp_text)
        src_windows = self.get_windows(src_text)

        if not susp_windows or not src_windows:
            return None

        src_texts = [w["text"] for w in src_windows]
        susp_texts = [w["text"] for w in susp_windows]

        src_embeddings = self.model.encode(src_texts, convert_to_tensor=True, show_progress_bar=False)
        susp_embeddings = self.model.encode(susp_texts, convert_to_tensor=True, show_progress_bar=False)

        # Vectorised feature extraction and structural matrix
        susp_feat_arr, _ = self._extract_features_batch(susp_windows)
        src_feat_arr, _ = self._extract_features_batch(src_windows)

        semantic_matrix = util.pytorch_cos_sim(susp_embeddings, src_embeddings).cpu().numpy()
        structural_matrix = self._compute_structural_matrix_vectorised(susp_feat_arr, src_feat_arr)

        return {
            "susp_windows": susp_windows,
            "src_windows": src_windows,
            "semantic_matrix": semantic_matrix,
            "structural_matrix": structural_matrix,
            "susp_text_length": len(susp_text),
        }

    def _smith_waterman_fast(self, hybrid_matrix, threshold, gap_penalty):
        """
        Smith-Waterman alignment with row-level NumPy vectorisation.
        The DP has row dependencies so full vectorisation is impossible,
        but the inner j-loop uses a bounded left-gap scan that skips
        most cells when gap_penalty is negative (typical: -0.5).
        """
        m, n = hybrid_matrix.shape
        score_matrix = np.zeros((m + 1, n + 1))

        # Precompute match/mismatch scores
        match_scores = np.where(hybrid_matrix >= threshold, hybrid_matrix, -1.0)

        # Maximum number of left-gap steps before score hits zero
        # (conservative bound: max possible score is ~1.0)
        max_gap_reach = int(1.0 / abs(gap_penalty)) + 2 if gap_penalty < 0 else n

        for i in range(1, m + 1):
            # Diagonal: score_matrix[i-1, 0..n-1] + match_scores[i-1, 0..n-1]
            diag = score_matrix[i - 1, :n] + match_scores[i - 1, :]

            # Up: score_matrix[i-1, 1..n] + gap_penalty
            up = score_matrix[i - 1, 1:] + gap_penalty

            # Start with max of diag, up, 0
            row = np.maximum(0, np.maximum(diag, up))

            # Left-gap propagation (sequential but bounded)
            # Each cell can only be improved by a left neighbor if
            # that neighbor's score + gap_penalty > current value.
            # With gap_penalty=-0.5, this dies after ~2 steps.
            prev = 0.0
            for j in range(n):
                left_val = prev + gap_penalty
                if left_val > row[j]:
                    row[j] = left_val
                prev = row[j]

            score_matrix[i, 1:] = row

        return score_matrix

    def detect_plagiarism_from_precomputed(
        self,
        pair_data,
        threshold=0.4,
        gap_penalty=-0.5,
        semantic_weight=None,
        structural_weight=None,
        chain_threshold=0.1,
        min_detection_length=0,
    ):
        """Runs alignment from cached pair data."""
        if pair_data is None:
            return []

        susp_windows = pair_data["susp_windows"]
        src_windows = pair_data["src_windows"]
        semantic_matrix = pair_data["semantic_matrix"]
        structural_matrix = pair_data["structural_matrix"]

        sem_weight = self.hybrid_scorer.w_sem if semantic_weight is None else semantic_weight
        str_weight = self.hybrid_scorer.w_str if structural_weight is None else structural_weight
        hybrid_matrix = (sem_weight * semantic_matrix) + (str_weight * structural_matrix)

        score_matrix = self._smith_waterman_fast(hybrid_matrix, threshold, gap_penalty)

        # Vectorised detection extraction
        m, n = hybrid_matrix.shape
        scores = score_matrix[1:, 1:]

        # Find cells above chain threshold that are local maxima (>= up and >= left)
        above_threshold = scores > chain_threshold
        ge_up = scores >= score_matrix[:m, 1:]
        ge_left = scores >= score_matrix[1:, :n]
        mask = above_threshold & ge_up & ge_left

        det_i, det_j = np.where(mask)

        detections = []
        for idx in range(len(det_i)):
            i, j = det_i[idx], det_j[idx]
            detections.append({
                "this_offset": susp_windows[i]["offset"],
                "this_length": susp_windows[i]["length"],
                "source_offset": src_windows[j]["offset"],
                "source_length": src_windows[j]["length"],
                "score": float(scores[i, j]),
            })

        merged = self.merge_detections(detections)

        # Filter out short detections (reduces false positives)
        if min_detection_length > 0:
            merged = [d for d in merged if d["this_length"] >= min_detection_length]

        return merged

    def detect_plagiarism(self, susp_text, src_text, threshold=0.4, gap_penalty=-0.5,
                          chain_threshold=0.1, min_detection_length=0):
        """Full detection pipeline for a single pair."""
        pair_data = self.precompute_pair_data(susp_text, src_text)
        return self.detect_plagiarism_from_precomputed(
            pair_data,
            threshold=threshold,
            gap_penalty=gap_penalty,
            chain_threshold=chain_threshold,
            min_detection_length=min_detection_length,
        )

    def merge_detections(self, detections):
        """Merge adjacent or overlapping detected windows."""
        if not detections:
            return []

        # Sort by offset for merging
        detections = sorted(detections, key=lambda x: x['this_offset'])

        merged = []
        current = detections[0].copy()

        for next_det in detections[1:]:
            if next_det['this_offset'] <= current['this_offset'] + current['this_length']:
                current['this_length'] = (next_det['this_offset'] + next_det['this_length']) - current['this_offset']
            else:
                merged.append(current)
                current = next_det.copy()

        merged.append(current)
        return merged

    def evaluate(self, detections, truths, text_length):
        """Character-level Precision, Recall, and F1."""
        pred_arr = np.zeros(text_length, dtype=np.int8)
        true_arr = np.zeros(text_length, dtype=np.int8)

        for d in detections:
            pred_arr[d['this_offset'] : d['this_offset'] + d['this_length']] = 1

        for t in truths:
            true_arr[t['this_offset'] : t['this_offset'] + t['this_length']] = 1

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
