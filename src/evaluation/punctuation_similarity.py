"""
📝 Punctuation Patrol! 📝

This module analyzes the use of punctuation across texts. It compares how
similar two texts are based on their punctuation patterns and frequencies.
Think of it as a style detective that focuses on those tiny but crucial marks
that give rhythm and structure to our writing.

Periods, commas, exclamations, oh my! This module will quantify how similar the
punctuation "personality" of two texts really is.

Warning: May cause you to notice punctuation everywhere you look! 🔍✨
"""

import re
import numpy as np
from collections import Counter
from typing import Tuple, Dict, List

# Define punctuation marks to analyze
PUNCTUATION_MARKS = ['.', ',', ';', ':', '!', '?', '-', '"', "'", '(', ')', '[', ']', '{', '}']

def extract_punctuation(text: str) -> List[str]:
    """
    Extract all punctuation marks from a text.
    
    Args:
        text: The input text
        
    Returns:
        List of punctuation marks found in the text
    """
    # This pattern matches all punctuation marks
    pattern = r'[.,;:!?\-"\'()\[\]{}]'
    return re.findall(pattern, text)

def get_punctuation_distribution(text: str) -> Dict[str, float]:
    """
    Calculate the distribution of punctuation marks in a text.
    
    Args:
        text: The input text
        
    Returns:
        Dictionary with punctuation marks as keys and their normalized frequencies as values
    """
    # Extract all punctuation
    punctuation_marks = extract_punctuation(text)
    
    # Count occurrences
    punct_counts = Counter(punctuation_marks)
    
    # Calculate total count
    total = sum(punct_counts.values()) if punct_counts else 1  # Avoid division by zero
    
    # Normalize frequencies
    punct_dist = {mark: punct_counts.get(mark, 0) / total for mark in PUNCTUATION_MARKS}
    
    # Add punctuation density (marks per word)
    word_count = len(text.split())
    if word_count > 0:
        punct_dist['density'] = len(punctuation_marks) / word_count
    else:
        punct_dist['density'] = 0
    
    return punct_dist

def punctuation_similarity(text1: str, text2: str) -> Tuple[float, Dict]:
    """
    Compare two texts based on their punctuation mark distributions.
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        
    Returns:
        Tuple of (similarity_score, details_dict) where:
        - similarity_score is a float between 0 and 1
        - details_dict contains the punctuation distributions and other statistics
    """
    # Get punctuation distributions
    dist1 = get_punctuation_distribution(text1)
    dist2 = get_punctuation_distribution(text2)
    
    # Extract features for comparison (all standard punctuation + density)
    features = PUNCTUATION_MARKS + ['density']
    
    # Create feature vectors
    vec1 = np.array([dist1[feature] for feature in features])
    vec2 = np.array([dist2[feature] for feature in features])
    
    # Calculate cosine similarity
    if np.linalg.norm(vec1) * np.linalg.norm(vec2) > 0:  # Avoid division by zero
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    else:
        similarity = 0.0 if np.any(vec1 != vec2) else 1.0
    
    # Prepare detailed output
    details = {
        "text1_stats": {
            "punct_distribution": dist1,
            "most_common": Counter(extract_punctuation(text1)).most_common(3),
            "punct_per_word": dist1['density']
        },
        "text2_stats": {
            "punct_distribution": dist2,
            "most_common": Counter(extract_punctuation(text2)).most_common(3),
            "punct_per_word": dist2['density']
        },
        "difference": {
            mark: abs(dist1[mark] - dist2[mark]) 
            for mark in features
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
        similarity_score, punct_details = punctuation_similarity(text1, text2)
        
        # Print results
        print(f"Text 1: {file1}")
        print(f"Text 2: {file2}")
        print(f"\nPunctuation Similarity score: {similarity_score:.4f}")
        print(f"\nPunctuation density in Text 1: {punct_details['text1_stats']['punct_per_word']:.4f} marks per word")
        print(f"Punctuation density in Text 2: {punct_details['text2_stats']['punct_per_word']:.4f} marks per word")
        print("\nMost common punctuation in Text 1:", punct_details['text1_stats']['most_common'])
        print("Most common punctuation in Text 2:", punct_details['text2_stats']['most_common'])
    except FileNotFoundError:
        print(f"Error: One or both of the files could not be found. Please check the file paths.")
    except Exception as e:
        print(f"Error: {str(e)}")
