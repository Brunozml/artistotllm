"""
🎭 Parts of Speech Party! 🎭

This module is your friendly neighborhood grammar detective! It analyzes text by looking at
Parts of Speech (POS) patterns - you know, those sneaky nouns, verbs, adjectives, and their friends.
Think of it as a linguistic DJ that creates a "word category remix" of your text and compares
how similar two texts are based on their grammatical vibes.

Warning: May cause unexpected appreciation for grammar rules! 📚✨
"""

import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
import numpy as np
from typing import Tuple, List, Set

# Download required NLTK data
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

def get_pos_distribution(text: str) -> dict:
    """
    Get the distribution of POS tags in the text.
    Returns a dictionary with POS tags as keys and their frequencies as values.
    """
    # Tokenize and get POS tags
    tokens = word_tokenize(text.lower())
    pos_tags = pos_tag(tokens)
    
    # Count POS tag frequencies
    pos_dist = {}
    for _, tag in pos_tags:
        pos_dist[tag] = pos_dist.get(tag, 0) + 1
    
    # Normalize frequencies
    total = sum(pos_dist.values())
    for tag in pos_dist:
        pos_dist[tag] = pos_dist[tag] / total
        
    return pos_dist

def pos_similarity(text1: str, text2: str) -> Tuple[float, dict]:
    """
    Compare two texts based on their POS tag distributions.
    Returns a similarity score and the POS distributions.
    """
    # Get POS distributions
    dist1 = get_pos_distribution(text1)
    dist2 = get_pos_distribution(text2)
    
    # Get all unique POS tags
    all_tags = set(dist1.keys()) | set(dist2.keys())
    
    # Calculate cosine similarity
    vec1 = np.array([dist1.get(tag, 0) for tag in all_tags])
    vec2 = np.array([dist2.get(tag, 0) for tag in all_tags])
    
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    return similarity, {"text1_pos": dist1, "text2_pos": dist2}

def read_file(filepath):
    """Read text from a file and return its contents"""
    with open(filepath, 'r') as f:
        return f.read()

if __name__ == "__main__":
    # Example texts
    # text1 = "The cat quickly jumped over the lazy dog."
    # text2 = "A dog slowly walked under the tired cat."
    # Read the files
    data_path = '/Users/brunozorrilla/Documents/GitHub/artistotllm/data/raw/'
    file1 = 'gpt_what_to_do.txt'
    file2 = 'hypewrite_what_to_do.txt'

    text1 = read_file(data_path + file1)
    text2 = read_file(data_path + file2)
    
    # Compare texts
    similarity_score, pos_distributions = pos_similarity(text1, text2)
    
    # Print results
    print(f"Text 1: {file1}")
    print(f"Text 2: {file2}")
    print(f"\nPOS Similarity score: {similarity_score:.2f}")
    # print("\nPOS Distribution Text 1:")
    # for pos, freq in pos_distributions["text1_pos"].items():
    #     print(f"{pos}: {freq:.2f}")
    # print("\nPOS Distribution Text 2:")
    # for pos, freq in pos_distributions["text2_pos"].items():
    #     print(f"{pos}: {freq:.2f}")
