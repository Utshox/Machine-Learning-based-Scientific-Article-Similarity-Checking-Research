import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string

class StructuralFeatureExtractor:
    """
    Extracts structural and stylistic features from text for similarity analysis.
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def extract_features(self, text):
        """
        Extracts a dictionary of features from the given text.
        """
        tokens = word_tokenize(text.lower())
        words = [t for t in tokens if t not in string.punctuation]
        sentences = sent_tokenize(text)
        
        # 1. Basic Counts
        word_count = len(words)
        char_count = len(text)
        sentence_count = len(sentences)
        avg_word_length = char_count / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        # 2. Punctuation and Stopwords
        punctuation_count = sum(1 for char in text if char in string.punctuation)
        stopword_count = sum(1 for word in words if word in self.stop_words)
        punctuation_ratio = punctuation_count / char_count if char_count > 0 else 0
        stopword_ratio = stopword_count / word_count if word_count > 0 else 0

        # 3. POS Tagging (Noun, Verb, Adjective ratios)
        pos_tags = nltk.pos_tag(words)
        noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
        verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
        adj_count = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
        
        noun_ratio = noun_count / word_count if word_count > 0 else 0
        verb_ratio = verb_count / word_count if word_count > 0 else 0
        adj_ratio = adj_count / word_count if word_count > 0 else 0

        return {
            'word_count': word_count,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'punctuation_ratio': punctuation_ratio,
            'stopword_ratio': stopword_ratio,
            'noun_ratio': noun_ratio,
            'verb_ratio': verb_ratio,
            'adj_ratio': adj_ratio
        }

    def get_feature_diff(self, features1, features2):
        """
        Calculate the absolute difference between two sets of features.
        Used to assess structural similarity.
        """
        diff = {}
        for key in features1:
            diff[key] = abs(features1[key] - features2[key])
        return diff
