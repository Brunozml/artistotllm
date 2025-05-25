"""
📏 Sentence Length Analyzer! 📏

This module analyzes the average sentence length of texts and compares them.
It provides insights into the typical sentence structure and complexity
of different texts, which can be useful for stylometric analysis.

Warning: May cause unexpected appreciation for punctuation! 📝✨
"""

import re
import numpy as np
from typing import Tuple, Dict, List

def calculate_sentence_stats(text: str) -> Dict[str, float]:
    """
    Calculate average sentence length and other sentence statistics.
    
    Args:
        text (str): The input text to analyze
        
    Returns:
        dict: Dictionary containing sentence statistics
    """
    # Clean and prepare text
    text = text.strip()

    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Count words in each sentence
    sentence_word_counts = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    
    # Calculate statistics
    avg_sentence_length = sum(sentence_word_counts) / len(sentences) if sentences else 0
    median_length = np.median(sentence_word_counts) if sentences else 0
    std_dev = np.std(sentence_word_counts) if sentences else 0
    
    return {
        "avg_sentence_length": avg_sentence_length,
        "median_sentence_length": median_length,
        "std_deviation": std_dev,
        "total_sentences": len(sentences),
        "sentence_lengths": sentence_word_counts
    }

def sentence_length_similarity(text1: str, text2: str) -> Tuple[float, Dict]:
    """
    Compare two texts based on their average sentence lengths.
    Returns a similarity score and the sentence statistics.
    
    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare
        
    Returns:
        Tuple containing:
        - similarity score (float): 1.0 means identical average sentence length, 
          closer to 0.0 means more different
        - dictionary with sentence statistics for both texts
    """
    # Get sentence statistics
    stats1 = calculate_sentence_stats(text1)
    stats2 = calculate_sentence_stats(text2)
    
    # Calculate similarity as the ratio of the shorter average to the longer one
    # This gives a value between 0 and 1, where 1 means identical average lengths
    avg1 = stats1["avg_sentence_length"]
    avg2 = stats2["avg_sentence_length"]
    
    if avg1 == 0 and avg2 == 0:  # Edge case: both texts have no sentences
        similarity = 1.0
    elif avg1 == 0 or avg2 == 0:  # Edge case: one text has no sentences
        similarity = 0.0
    else:
        similarity = min(avg1, avg2) / max(avg1, avg2)
    
    return similarity, {
        "text1_stats": stats1,
        "text2_stats": stats2,
        "difference": avg2 - avg1  # positive if text2 has longer sentences
    }

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
    similarity_score, stats = sentence_length_similarity(text1, text2)
    
    # Print results
    print(f"Text 1: {file1}")
    print(f"   Average sentence length: {stats['text1_stats']['avg_sentence_length']:.1f} words")
    print(f"   Total sentences: {stats['text1_stats']['total_sentences']}")
    
    print(f"\nText 2: {file2}")
    print(f"   Average sentence length: {stats['text2_stats']['avg_sentence_length']:.1f} words")
    print(f"   Total sentences: {stats['text2_stats']['total_sentences']}")
    
    print(f"\nSentence Length Similarity score: {similarity_score:.2f}")
    print(f"Difference (Text 2 - Text 1): {stats['difference']:+.1f} words")
