import os
import xml.etree.ElementTree as ET

class PANDataLoader:
    """
    Handles loading of PAN dataset files (txt and XML truths).
    """
    def __init__(self, base_path):
        self.base_path = base_path
        self.susp_dir = os.path.join(base_path, 'susp')
        self.src_dir = os.path.join(base_path, 'src')
        self.pairs_file = os.path.join(base_path, 'pairs')

    def get_pairs(self):
        """Returns a list of (suspicious_file, source_file) tuples."""
        pairs = []
        if not os.path.exists(self.pairs_file):
            return pairs
        with open(self.pairs_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
        return pairs

    def load_text(self, filename, is_suspicious=True):
        """Loads the content of a text file."""
        folder = self.susp_dir if is_suspicious else self.src_dir
        path = os.path.join(folder, filename)
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_truth(self, susp_filename):
        """Parses the XML truth file for a suspicious document."""
        xml_filename = susp_filename.replace('.txt', '.xml')
        path = os.path.join(self.susp_dir, xml_filename)
        
        truths = []
        if not os.path.exists(path):
            return truths

        tree = ET.parse(path)
        root = tree.getroot()
        
        for feature in root.findall('feature'):
            if feature.get('name') == 'plagiarism':
                truths.append({
                    'this_offset': int(feature.get('this_offset')),
                    'this_length': int(feature.get('this_length')),
                    'source_reference': feature.get('source_reference'),
                    'source_offset': int(feature.get('source_offset')),
                    'source_length': int(feature.get('source_length'))
                })
        return truths
