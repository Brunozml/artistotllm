"""
📝 Text Similarity Analyzer! 📝

This module analyzes how similar two texts are based on their content.
It provides methods to compute text similarity using various metrics
such as cosine similarity on TF-IDF vectors.

Warning: May cause unexpected appreciation for the beauty of text patterns! 📚✨
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Tuple, List, Dict

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text: str) -> str:
    """
    Preprocess text by lowercasing, removing stop words and punctuation
    """
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(filtered_tokens)

def get_text_vectors(text1: str, text2: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert texts to TF-IDF vectors
    """
    # Preprocess texts
    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)
    
    # Vectorize
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_text1, processed_text2])
    
    # Get vectors
    vector1 = tfidf_matrix[0]
    vector2 = tfidf_matrix[1]
    
    return vector1, vector2

def text_similarity(text1: str, text2: str) -> Tuple[float, Dict]:
    """
    Compare two texts based on their content using cosine similarity.
    Returns a similarity score and a dictionary with additional metrics.
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        
    Returns:
        Tuple containing:
        - similarity score (float)
        - dictionary with additional metrics
    """
    # Get TF-IDF vectors
    vector1, vector2 = get_text_vectors(text1, text2)
    
    # Calculate cosine similarity
    similarity_score = cosine_similarity(vector1, vector2)[0][0]
    
    # Calculate additional metrics
    word_count1 = len(text1.split())
    word_count2 = len(text2.split())
    
    metrics = {
        "word_count_text1": word_count1,
        "word_count_text2": word_count2,
        "word_count_ratio": min(word_count1, word_count2) / max(word_count1, word_count2)
    }
    
    return similarity_score, metrics

def read_file(filepath):
    """Read text from a file and return its contents"""
    with open(filepath, 'r') as f:
        return f.read()

if __name__ == "__main__":
    # Read the files
    file1 = 'data/raw/pg_essays/what_to_do.txt'	
    file2 = 'data/raw/pg_llama/what_to_do.txt'

    text1 = read_file(file1)
    text2 = read_file(file2)
    
    # Compare texts
    similarity_score, metrics = text_similarity(text1, text2)
    
    # Print results
    print(f"Text 1: {file1}")
    print(f"Text 2: {file2}")
    print(f"\nText Similarity score: {similarity_score:.4f}")
    print(f"Word count text 1: {metrics['word_count_text1']}")
    print(f"Word count text 2: {metrics['word_count_text2']}")
    print(f"Word count ratio: {metrics['word_count_ratio']:.2f}")
