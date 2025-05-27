"""
🛑 Stopwords Society! 🛑

This module analyzes the frequency and distribution of stopwords in texts.
While many analyses remove stopwords, this one celebrates them! The humble 'the',
'and', 'but' - these little words reveal subtle patterns of style and rhythm.
Think of it as a magnifying glass for the "invisible" words that give your text
its natural flow and cadence.

Warning: May cause you to suddenly notice just how many times you use 'the'! 📚✨
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from collections import Counter
from typing import Tuple, Dict, List

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Get the list of English stopwords
ENGLISH_STOPWORDS = set(stopwords.words('english'))

def extract_stopwords(text: str) -> List[str]:
    """
    Extract all stopwords from a text.
    
    Args:
        text (str): The input text
        
    Returns:
        List[str]: List of stopwords found in the text
    """
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Filter to keep only stopwords
    return [token for token in tokens if token in ENGLISH_STOPWORDS]

def get_stopword_distribution(text: str) -> Dict[str, float]:
    """
    Calculate the distribution of stopwords in a text.
    
    Args:
        text (str): The input text
        
    Returns:
        Dict[str, float]: Dictionary with stopwords as keys and their normalized frequencies as values
    """
    # Extract all stopwords
    stopword_tokens = extract_stopwords(text)
    
    # Count occurrences
    stopword_counts = Counter(stopword_tokens)
    
    # Calculate total count
    total = sum(stopword_counts.values()) if stopword_counts else 1  # Avoid division by zero
    
    # Normalize frequencies
    stopword_dist = {word: stopword_counts.get(word, 0) / total for word in ENGLISH_STOPWORDS}
    
    # Add stopword density (stopwords per total words)
    word_count = len(word_tokenize(text.lower()))
    if word_count > 0:
        stopword_dist['density'] = len(stopword_tokens) / word_count
    else:
        stopword_dist['density'] = 0
    
    return stopword_dist

def stopwords_similarity(text1: str, text2: str) -> Tuple[float, Dict]:
    """
    Compare two texts based on their stopword distributions.
    
    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare
        
    Returns:
        Tuple[float, Dict]: Tuple of (similarity_score, details_dict) where:
        - similarity_score is a float between 0 and 1
        - details_dict contains the stopword distributions and other statistics
    """
    # Get stopword distributions
    dist1 = get_stopword_distribution(text1)
    dist2 = get_stopword_distribution(text2)
    
    # Extract features for comparison (all standard stopwords + density)
    features = list(ENGLISH_STOPWORDS) + ['density']
    
    # Create feature vectors
    vec1 = np.array([dist1[feature] for feature in features])
    vec2 = np.array([dist2[feature] for feature in features])
    
    # Calculate cosine similarity
    if np.linalg.norm(vec1) * np.linalg.norm(vec2) > 0:  # Avoid division by zero
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    else:
        similarity = 0.0 if np.any(vec1 != vec2) else 1.0
    
    # Get top stopwords for each text
    text1_stopwords = extract_stopwords(text1)
    text2_stopwords = extract_stopwords(text2)
    top_stopwords1 = Counter(text1_stopwords).most_common(10)
    top_stopwords2 = Counter(text2_stopwords).most_common(10)
    
    # Calculate overlap in top 10 most frequent stopwords
    top10_set1 = {word for word, _ in top_stopwords1}
    top10_set2 = {word for word, _ in top_stopwords2}
    overlap = len(top10_set1.intersection(top10_set2))
    
    # Prepare detailed output
    details = {
        "text1_stats": {
            "stopword_distribution": {key: value for key, value in dist1.items() if key in list(ENGLISH_STOPWORDS)[:20] or key == 'density'},
            "most_common": top_stopwords1,
            "stopword_density": dist1['density']
        },
        "text2_stats": {
            "stopword_distribution": {key: value for key, value in dist2.items() if key in list(ENGLISH_STOPWORDS)[:20] or key == 'density'},
            "most_common": top_stopwords2,
            "stopword_density": dist2['density']
        },
        "top10_overlap": overlap,
        "top10_overlap_percent": overlap * 10,
        "difference": {
            word: abs(dist1[word] - dist2[word]) 
            for word in list(ENGLISH_STOPWORDS)[:20] + ['density']
        }
    }
    
    return similarity, details

def read_file(filepath):
    """Read text from a file and return its contents"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

if __name__ == "__main__":
    # Example usage when run directly
    file1 = 'data/raw/pg_essays/what_to_do.txt'	
    file2 = 'data/raw/pg_llama/what_to_do.txt'

    try:
        text1 = read_file(file1)
        text2 = read_file(file2)
        
        # Compare texts
        similarity_score, stopword_details = stopwords_similarity(text1, text2)
        
        # Print results
        print(f"Text 1: {file1}")
        print(f"Text 2: {file2}")
        print(f"\nStopword Similarity score: {similarity_score:.4f}")
        print(f"\nStopword density in Text 1: {stopword_details['text1_stats']['stopword_density']:.4f} stopwords per word")
        print(f"Stopword density in Text 2: {stopword_details['text2_stats']['stopword_density']:.4f} stopwords per word")
        print("\nTop 10 stopwords in Text 1:", stopword_details['text1_stats']['most_common'])
        print()
        print()
        print("Top 10 stopwords in Text 2:", stopword_details['text2_stats']['most_common'])
        print()
        print(f"Overlap in top 10 stopwords: {stopword_details['top10_overlap']} words ({stopword_details['top10_overlap_percent']}%)")
    except FileNotFoundError:
        print(f"Error: One or both of the files could not be found. Please check the file paths.")
    except Exception as e:
        print(f"Error: {str(e)}")
