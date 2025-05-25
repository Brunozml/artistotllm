"""
Sample script to demonstrate the pos_similarity and text_similarity modules.
This script analyzes similarity between pairs of text files.
"""

import os
import sys
import json
from datetime import datetime

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.pos_similarity import pos_similarity, read_file
from evaluation.text_similarity import text_similarity

def analyze_file_pairs(pairs_list, output_dir=None):
    """
    Analyze POS and text similarity between pairs of files.
    
    Args:
        pairs_list: List of tuples containing (file1_path, file2_path, pair_name)
        output_dir: Directory to save results (optional)
    """
    results = []
    
    for file1_path, file2_path, pair_name in pairs_list:
        print(f"\nAnalyzing pair: {pair_name}")
        
        try:
            # Read files
            text1 = read_file(file1_path)
            text2 = read_file(file2_path)
            
            # Get POS similarity score
            pos_similarity_score, pos_distributions = pos_similarity(text1, text2)
            
            # Get text similarity score
            text_similarity_score, text_metrics = text_similarity(text1, text2)
            
            # Print results
            print(f"File 1: {file1_path}")
            print(f"File 2: {file2_path}")
            print(f"POS Similarity score: {pos_similarity_score:.4f}")
            print(f"Text Similarity score: {text_similarity_score:.4f}")
            
            # Save results
            pair_result = {
                "pair_name": pair_name,
                "file1": file1_path,
                "file2": file2_path,
                "pos_similarity_score": pos_similarity_score,
                "pos_distributions": pos_distributions,
                "text_similarity_score": text_similarity_score,
                "text_metrics": text_metrics
            }
            results.append(pair_result)
            
        except Exception as e:
            print(f"Error analyzing pair {pair_name}: {e}")
    
    # Save results to file if output directory is provided
    if output_dir and results:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"similarity_results_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    # Define file pairs to analyze
    pairs = [
        ('data/raw/pg_essays/what_to_do.txt', 'data/raw/pg_llama/what_to_do.txt', 'what_to_do'),
        # Add more pairs as needed
        # ('path/to/file1.txt', 'path/to/file2.txt', 'pair_name'),
    ]
    
    # Output directory for results
    results_dir = 'results/similarity_analysis'
    
    # Run analysis
    analyze_file_pairs(pairs, results_dir)
