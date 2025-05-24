"""
🎯 N-gram Navigator: Your Text's Best Friend! 🎯

Welcome to the N-gram similarity analyzer, where we slice and dice text into delicious
bite-sized chunks (n-grams) and compare them using the mighty TF-IDF powers! It's like
a word sandwich detector that tells you how similar two texts are based on their
ingredient combinations.

Pro tip: Works best with a cup of coffee and a sense of humor! ☕️🔍
"""

from collections import Counter
from typing import List, Tuple, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# TODO: remove stop words using nltk 


def get_ngrams(text: str, n: int) -> List[str]:
    """Generate n-grams from text"""
    words = text.lower().split()
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def calculate_tfidf(texts: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Calculate TF-IDF vectors for the texts"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix.toarray(), vectorizer.get_feature_names_out()

def ngram_similarity(text1: str, text2: str, n: int = 2) -> Tuple[float, Dict]:
    """
    Compare two texts using n-gram overlap with TF-IDF weights.
    Returns a similarity score and the n-grams with their TF-IDF weights.
    """
    # Get n-grams
    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)
    
    # Join n-grams back to text for TF-IDF calculation
    text1_ngrams = ' '.join(ngrams1)
    text2_ngrams = ' '.join(ngrams2)
    
    # Calculate TF-IDF
    tfidf_matrix, feature_names = calculate_tfidf([text1_ngrams, text2_ngrams])
    
    # Calculate cosine similarity
    similarity = np.dot(tfidf_matrix[0], tfidf_matrix[1]) / (
        np.linalg.norm(tfidf_matrix[0]) * np.linalg.norm(tfidf_matrix[1])
    )
    
    # Get top weighted n-grams for each text
    def get_top_ngrams(tfidf_vector, feature_names, top_n=5):
        indices = np.argsort(tfidf_vector)[-top_n:]
        return {feature_names[i]: float(tfidf_vector[i]) for i in indices if tfidf_vector[i] > 0}
    
    return similarity, {
        "text1_top_ngrams": get_top_ngrams(tfidf_matrix[0], feature_names),
        "text2_top_ngrams": get_top_ngrams(tfidf_matrix[1], feature_names)
    }

def read_file(filepath):
    """Read text from a file and return its contents"""
    with open(filepath, 'r') as f:
        return f.read()

if __name__ == "__main__":
    # Example texts
    # text1 = "The cat quickly jumped over the lazy dog."
    # text2 = "A dog slowly walked under the tired cat."
    # Read the files
    file1 = 'data/raw/pg_essays/what_to_do.txt'
    file2 = 'data/raw/pg_llama/what_to_do.txt'

    text1 = read_file(file1)
    text2 = read_file(file2)
    # Compare texts
    similarity_score, gram_info = ngram_similarity(text1, text2)
    
    # Print results
    print(f"Text 1: {file1}")
    print(f"Text 2: {file2}")

    print(f"\nN-gram Similarity score: {similarity_score:.2f}")
    print("\nTop weighted n-grams in Text 1:")
    for ngram, weight in gram_info["text1_top_ngrams"].items():
        print(f"'{ngram}': {weight:.3f}")
    print("\nTop weighted n-grams in Text 2:")
    for ngram, weight in gram_info["text2_top_ngrams"].items():
        print(f"'{ngram}': {weight:.3f}")
