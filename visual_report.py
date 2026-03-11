import logging
from sentence_transformers import SentenceTransformer
from data_loader import PANDataLoader
from features import StructuralFeatureExtractor
from scoring import HybridScorer
from evaluator import PANEvaluator

class VisualReporter:
    """
    Highlights detected plagiarism in the console.
    """
    def __init__(self):
        # ANSI Color Codes
        self.RED = '\033[91m'
        self.GREEN = '\033[92m'
        self.BOLD = '\033[1m'
        self.END = '\033[0m'

    def generate_report(self, susp_text, detections, truths):
        """Prints the suspicious document with highlighted plagiarism."""
        print("\n" + "="*80)
        print(f"{self.BOLD}PLAGIARISM VISUAL REPORT (PAN 2025 MOCK DATA){self.END}")
        print("="*80 + "\n")

        # 1. Show the "Ground Truth" vs "Detected" segments
        print(f"{self.GREEN}■ Ground Truth Plagiarism{self.END}")
        print(f"{self.RED}■ Model Detected Plagiarism{self.END}\n")

        # Sort detections by offset
        detections = sorted(detections, key=lambda x: x['this_offset'])
        
        # Build the highlighted text
        output_text = ""
        last_idx = 0
        
        # Simple character-level flagging (Red for detected)
        for det in detections:
            # Add normal text before detection
            output_text += susp_text[last_idx : det['this_offset']]
            # Add highlighted text
            flagged_text = susp_text[det['this_offset'] : det['this_offset'] + det['this_length']]
            output_text += f"{self.RED}{self.BOLD}{flagged_text}{self.END}"
            last_idx = det['this_offset'] + det['this_length']
            
        output_text += susp_text[last_idx:]
        print(output_text)
        print("\n" + "="*80)

def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    feature_extractor = StructuralFeatureExtractor()
    hybrid_scorer = HybridScorer(semantic_weight=0.7, structural_weight=0.3)
    evaluator = PANEvaluator(model, feature_extractor, hybrid_scorer, window_size=150, step_size=25)
    
    loader = PANDataLoader("/Users/shinthiya.promi/Desktop/MS_THESIS/implementation/data/pan2025_mock")
    reporter = VisualReporter()
    
    pair = loader.get_pairs()[0]
    susp_text = loader.load_text(pair[0], is_suspicious=True)
    src_text = loader.load_text(pair[1], is_suspicious=False)
    truths = loader.load_truth(pair[0])
    
    # Run detection with the tuned threshold (0.20-0.30)
    detections = evaluator.detect_plagiarism(susp_text, src_text, threshold=0.25)
    
    reporter.generate_report(susp_text, detections, truths)

if __name__ == "__main__":
    main()
