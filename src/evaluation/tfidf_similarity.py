"""
🔠 TF-IDF Textual Tapestry! 🔠

This module analyzes the semantic similarity of texts using TF-IDF (Term Frequency-Inverse Document Frequency).
It compares how similar two texts are based on their weighted word and n-gram patterns, accounting
for both common and distinctive terms. Think of it as a content detective that finds meaning
beyond just word count, revealing how truly similar texts are in what they're actually saying.

Warning: May cause unexpected appreciation for how uniquely we express similar ideas! 📊✨
"""

import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import Tuple, Dict, List, Any, Union

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text: str) -> str:
    """
    Preprocess text by lowercasing and normalizing whitespace.
    
    Args:
        text (str): The input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    # Simple preprocessing - sklearn's TfidfVectorizer will handle tokenization
    return text.lower()

def extract_top_features(tfidf_matrix: np.ndarray, feature_names: np.ndarray, top_n: int = 10) -> Dict[str, float]:
    """
    Extract top weighted features from TF-IDF matrix for a document.
    
    Args:
        tfidf_matrix (np.ndarray): Row of the TF-IDF matrix for a document
        feature_names (np.ndarray): Array of feature names
        top_n (int): Number of top features to extract
        
    Returns:
        Dict[str, float]: Dictionary of top features with their weights
    """
    # Get indices of top weighted features
    top_indices = np.argsort(tfidf_matrix)[::-1][:top_n]
    
    # Create a dictionary of feature names and their weights
    top_features = {feature_names[i]: float(tfidf_matrix[i]) for i in top_indices if tfidf_matrix[i] > 0}
    
    return top_features

def analyze_unique_features(weights1: np.ndarray, weights2: np.ndarray, feature_names: np.ndarray, top_n: int = 5) -> Dict[str, List[str]]:
    """
    Find features that are unique to each text.
    
    Args:
        weights1 (np.ndarray): TF-IDF weights for first text
        weights2 (np.ndarray): TF-IDF weights for second text
        feature_names (np.ndarray): Array of feature names
        top_n (int): Number of top unique features to extract
        
    Returns:
        Dict[str, List[str]]: Dictionary with unique features for each text
    """
    # Features present in text1 but not in text2
    unique_to_text1 = [(i, weights1[i]) for i in range(len(weights1)) if weights1[i] > 0 and weights2[i] == 0]
    unique_to_text1.sort(key=lambda x: x[1], reverse=True)
    
    # Features present in text2 but not in text1
    unique_to_text2 = [(i, weights2[i]) for i in range(len(weights2)) if weights2[i] > 0 and weights1[i] == 0]
    unique_to_text2.sort(key=lambda x: x[1], reverse=True)
    
    # Extract feature names
    unique_features_text1 = [feature_names[idx] for idx, _ in unique_to_text1[:top_n]]
    unique_features_text2 = [feature_names[idx] for idx, _ in unique_to_text2[:top_n]]
    
    return {
        "unique_to_text1": unique_features_text1,
        "unique_to_text2": unique_features_text2
    }

def tfidf_similarity(text1: str, text2: str, ngram_range: Tuple[int, int] = (1, 3), remove_stopwords: bool = True) -> Tuple[float, Dict[str, Any]]:
    """
    Compare two texts using TF-IDF vectors across multiple n-gram ranges.
    
    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare
        ngram_range (tuple): Range of n-gram sizes to include (min, max)
        remove_stopwords (bool): Whether to remove stopwords
        
    Returns:
        Tuple[float, Dict]: A tuple containing:
            - A similarity score between 0-1
            - A dictionary with detailed metrics
    """
    # Preprocess texts
    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)
    
    # Process stopwords if needed
    stop_words = 'english' if remove_stopwords else None
    
    # Create vectorizer for the specified n-gram range
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range, 
        stop_words=stop_words,
        sublinear_tf=True  # Apply sublinear scaling to term frequencies
    )
    
    # Fit and transform both texts
    try:
        tfidf_matrix = vectorizer.fit_transform([processed_text1, processed_text2])
        feature_names = vectorizer.get_feature_names_out()
    except ValueError as e:
        # Handle empty corpus or other vectorization errors
        return 0.0, {"error": str(e)}
    
    # Calculate cosine similarity
    if tfidf_matrix.shape[1] > 0:  # Ensure we have features
        # Convert sparse matrix to dense for simpler operations
        dense_matrix = tfidf_matrix.toarray()
        # Calculate cosine similarity
        dot_product = np.dot(dense_matrix[0], dense_matrix[1])
        norm_text1 = np.linalg.norm(dense_matrix[0])
        norm_text2 = np.linalg.norm(dense_matrix[1])
        
        if norm_text1 > 0 and norm_text2 > 0:  # Avoid division by zero
            similarity = dot_product / (norm_text1 * norm_text2)
        else:
            similarity = 0.0
    else:
        similarity = 0.0
    
    # Extract top features for each text
    if tfidf_matrix.shape[1] > 0:
        dense_matrix = tfidf_matrix.toarray()
        top_features1 = extract_top_features(dense_matrix[0], feature_names)
        top_features2 = extract_top_features(dense_matrix[1], feature_names)
        
        # Analyze unique features
        unique_features = analyze_unique_features(dense_matrix[0], dense_matrix[1], feature_names)
        
        # Get n-gram level statistics
        ngram_stats = {}
        for n in range(ngram_range[0], ngram_range[1] + 1):
            # Filter features by n-gram length
            n_gram_features = [f for f in feature_names if len(f.split()) == n]
            if n_gram_features:
                ngram_stats[f"{n}-gram"] = len(n_gram_features)
    else:
        top_features1 = {}
        top_features2 = {}
        unique_features = {"unique_to_text1": [], "unique_to_text2": []}
        ngram_stats = {}
    
    # Prepare detailed output
    details = {
        "similarity": similarity,
        "ngram_range": ngram_range,
        "stopwords_removed": remove_stopwords,
        "top_features_text1": top_features1,
        "top_features_text2": top_features2,
        "unique_features": unique_features,
        "ngram_stats": ngram_stats,
        "vectorizer_vocabulary_size": len(feature_names) if feature_names is not None else 0
    }
    
    return similarity, details

def read_file(filepath):
    """Read text from a file and return its contents"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

if __name__ == "__main__":
    # Example usage when run directly
    file1 = 'data/PG_sample/processed/a_plan_for_spam.txt'
    file2 = 'data/PG_sample/generated/a_plan_for_spam_generated.txt'

    try:
        text1 = read_file(file1)
        text2 = read_file(file2)
        
        # Compare texts with different n-gram ranges
        # 1. Unigrams only
        uni_sim, uni_details = tfidf_similarity(text1, text2, ngram_range=(1, 1))
        # 2. Unigrams and bigrams
        bi_sim, bi_details = tfidf_similarity(text1, text2, ngram_range=(1, 2))
        # 3. Unigrams, bigrams, and trigrams
        tri_sim, tri_details = tfidf_similarity(text1, text2, ngram_range=(1, 3))
        
        # Print results
        print(f"Text 1: {file1}")
        print(f"Text 2: {file2}")
        
        print(f"\nTF-IDF Similarity (unigrams): {uni_sim:.4f}")
        print(f"TF-IDF Similarity (unigrams + bigrams): {bi_sim:.4f}")
        print(f"TF-IDF Similarity (unigrams + bigrams + trigrams): {tri_sim:.4f}")
        
        print("\nTop features in Text 1:")
        for feature, weight in tri_details["top_features_text1"].items():
            print(f"'{feature}': {weight:.4f}")
        
        print("\nTop features in Text 2:")
        for feature, weight in tri_details["top_features_text2"].items():
            print(f"'{feature}': {weight:.4f}")
        
        print("\nUnique features in Text 1:")
        for feature in tri_details["unique_features"]["unique_to_text1"]:
            print(f"'{feature}'")
        
        print("\nUnique features in Text 2:")
        for feature in tri_details["unique_features"]["unique_to_text2"]:
            print(f"'{feature}'")
            
    except FileNotFoundError:
        print(f"Error: One or both of the files could not be found. Please check the file paths.")
    except Exception as e:
        print(f"Error: {str(e)}")
