"""
📊 Type-Token Ratio Symphony! 📊

This module analyzes the vocabulary richness of texts through Type-Token Ratio (TTR) metrics.
It examines the variety of unique words used relative to total words, revealing the lexical
diversity patterns that give each text its distinctive voice. Think of it as a vocabulary
detective that measures how repetitive or diverse a text's word usage really is.

Warning: May cause unexpected appreciation for linguistic variety! 📝✨
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from typing import Tuple, Dict, List

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text: str, remove_stopwords: bool = False) -> List[str]:
    """
    Preprocess text by tokenizing, lowercasing, and optionally removing stopwords.
    
    Args:
        text (str): The input text to preprocess
        remove_stopwords (bool): Whether to remove stopwords
        
    Returns:
        List[str]: Preprocessed tokens
    """
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove non-alphabetic tokens
    tokens = [token for token in tokens if token.isalpha()]
    
    # Optionally remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    return tokens

def calculate_ttr(tokens: List[str]) -> float:
    """
    Calculate the Type-Token Ratio (TTR).
    
    Args:
        tokens (List[str]): List of tokens from the text
        
    Returns:
        float: The Type-Token Ratio value
    """
    if not tokens:
        return 0
    
    n_types = len(set(tokens))  # Number of unique words
    n_tokens = len(tokens)      # Total number of words
    
    return n_types / n_tokens

def moving_average_ttr(tokens: List[str], window_size: int = 100) -> float:
    """
    Calculate Moving-Average Type-Token Ratio (MATTR).
    
    Args:
        tokens (List[str]): List of tokens from the text
        window_size (int): Size of the sliding window
        
    Returns:
        float: The MATTR value
    """
    if len(tokens) < window_size:
        return calculate_ttr(tokens)
    
    # Calculate TTR for each window and take the average
    ttrs = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i+window_size]
        ttrs.append(calculate_ttr(window))
    
    return sum(ttrs) / len(ttrs)

def mtld(tokens: List[str], threshold: float = 0.72) -> float:
    """
    Calculate Measure of Textual Lexical Diversity (MTLD).
    
    Args:
        tokens (List[str]): List of tokens from the text
        threshold (float): The TTR threshold for factor count
        
    Returns:
        float: The MTLD value
    """
    if len(tokens) < 50:  # Too short for reliable MTLD
        return 0
    
    def mtld_pass(tokens, threshold):
        # Forward pass
        factors = 0
        types_so_far = set()
        token_count = 0
        
        for token in tokens:
            token_count += 1
            types_so_far.add(token)
            ttr = len(types_so_far) / token_count
            
            if ttr <= threshold:
                factors += 1
                types_so_far = set()
                token_count = 0
        
        if token_count > 0:
            ttr = len(types_so_far) / token_count
            partial_factor = (1 - ttr) / (1 - threshold)
            factors += partial_factor
        
        return len(tokens) / factors if factors > 0 else 0
    
    # Calculate MTLD as the average of forward and backward passes
    forward = mtld_pass(tokens, threshold)
    backward = mtld_pass(tokens[::-1], threshold)
    
    return (forward + backward) / 2

def ttr_similarity(text1: str, text2: str, include_stopwords: bool = True) -> Tuple[float, Dict]:
    """
    Compare two texts based on their lexical diversity (TTR) metrics.
    
    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare
        include_stopwords (bool): Whether to include stopwords in the analysis
        
    Returns:
        Tuple[float, Dict]: A tuple containing:
            - A similarity score between 0-1
            - A dictionary with detailed metrics
    """
    # Preprocess texts
    tokens1 = preprocess_text(text1, remove_stopwords=not include_stopwords)
    tokens2 = preprocess_text(text2, remove_stopwords=not include_stopwords)
    
    # Calculate basic TTR for both texts
    ttr1 = calculate_ttr(tokens1)
    ttr2 = calculate_ttr(tokens2)
    
    # Calculate MATTR for both texts
    mattr1 = moving_average_ttr(tokens1)
    mattr2 = moving_average_ttr(tokens2)
    
    # Calculate MTLD for both texts
    mtld1 = mtld(tokens1)
    mtld2 = mtld(tokens2)
    
    # Calculate similarity scores (1 - normalized absolute difference)
    ttr_sim = 1 - abs(ttr1 - ttr2) / max(ttr1, ttr2) if max(ttr1, ttr2) > 0 else 1
    mattr_sim = 1 - abs(mattr1 - mattr2) / max(mattr1, mattr2) if max(mattr1, mattr2) > 0 else 1
    mtld_sim = 1 - abs(mtld1 - mtld2) / max(mtld1, mtld2) if max(mtld1, mtld2) > 0 else 1
    
    # Calculate overall similarity (weighted average)
    overall_sim = (ttr_sim * 0.3) + (mattr_sim * 0.4) + (mtld_sim * 0.3)
    
    # Prepare detailed output
    details = {
        "ttr": {
            "text1": ttr1, 
            "text2": ttr2, 
            "similarity": ttr_sim
        },
        "mattr": {
            "text1": mattr1, 
            "text2": mattr2, 
            "similarity": mattr_sim
        },
        "mtld": {
            "text1": mtld1, 
            "text2": mtld2, 
            "similarity": mtld_sim
        },
        "text1_stats": {
            "tokens": len(tokens1), 
            "unique_tokens": len(set(tokens1)),
            "lexical_density": len(set(tokens1)) / len(tokens1) if tokens1 else 0
        },
        "text2_stats": {
            "tokens": len(tokens2), 
            "unique_tokens": len(set(tokens2)),
            "lexical_density": len(set(tokens2)) / len(tokens2) if tokens2 else 0
        }
    }
    
    return overall_sim, details

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
        
        # Compare texts
        similarity_score, ttr_details = ttr_similarity(text1, text2)
        
        # Print results
        print(f"Text 1: {file1}")
        print(f"Text 2: {file2}")
        print(f"\nLexical Diversity Similarity score: {similarity_score:.4f}")
        
        print("\nDetailed Metrics:")
        print(f"Type-Token Ratio: {ttr_details['ttr']['similarity']:.4f}")
        print(f"Moving-Average TTR: {ttr_details['mattr']['similarity']:.4f}")
        print(f"MTLD: {ttr_details['mtld']['similarity']:.4f}")
        
        print("\nText Statistics:")
        print(f"Original Text: {ttr_details['text1_stats']['tokens']} tokens, "
              f"{ttr_details['text1_stats']['unique_tokens']} unique tokens")
        print(f"Generated Text: {ttr_details['text2_stats']['tokens']} tokens, "
              f"{ttr_details['text2_stats']['unique_tokens']} unique tokens")
        
    except FileNotFoundError:
        print(f"Error: One or both of the files could not be found. Please check the file paths.")
    except Exception as e:
        print(f"Error: {str(e)}")
