"""
Script to analyze and compare original essays with their generated counterparts.
This script calculates text and POS similarity metrics and stores results in a CSV file.
"""

import os
import sys
import csv
import glob
from datetime import datetime
import pandas as pd

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.pos_similarity import pos_similarity, read_file
from evaluation.text_similarity import text_similarity
from utils.text_splitter import split_text

# Set file paths
ORIGINAL_DIR = "data/PG_sample/processed"
GENERATED_DIR = "data/PG_sample/generated"
OUTPUT_DIR = "results/essay_analysis"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"essay_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

def extract_title_from_filename(filename):
    """Extract the essay title from the filename"""
    return os.path.basename(filename).replace('.txt', '')

def get_essay_pairs():
    """Find all essay pairs (original and generated versions)"""
    pairs = []
    
    # Get all original essay files
    original_files = glob.glob(os.path.join(ORIGINAL_DIR, "*.txt"))
    
    for original_file in original_files:
        title = extract_title_from_filename(original_file)
        generated_file = os.path.join(GENERATED_DIR, f"{title}_generated.txt")
        
        # Check if the generated file exists
        if os.path.exists(generated_file):
            pairs.append((title, original_file, generated_file))
    
    return pairs

def analyze_essays():
    """Analyze and compare all essay pairs"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create CSV file and write header
    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        fieldnames = [
            "Author",
            "Text_Length",
            "Path_Full_text",
            "First_500",
            "Second_500",
            "Model",
            "Path_generated_500",
            "Generated_500",
            "True_500",
            "text_similarity_LLM",
            "text_similarity_ORIGINAL",
            "pos_similarity_LLM",
            "pos_similarity_ORIGINAL",
            "date",
            "comments"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Get all essay pairs
        essay_pairs = get_essay_pairs()
        print(f"Found {len(essay_pairs)} essay pairs to analyze")
        
        # Process each pair
        for title, original_path, generated_path in essay_pairs:
            print(f"\nAnalyzing essay: {title}")
            
            try:
                # Read the original and generated texts
                original_text = read_file(original_path)
                generated_text = read_file(generated_path)
                
                # Split the original text into segments
                first_500, second_500, rest = split_text(original_text)
                
                # Calculate text lengths
                original_length = len(original_text.split())
                generated_length = len(generated_text.split())
                
                # Calculate similarities
                # 1. First 500 words of original vs generated
                text_sim_llm, _ = text_similarity(first_500, generated_text)
                pos_sim_llm, _ = pos_similarity(first_500, generated_text)
                
                # 2. First 500 words vs second 500 words of original
                text_sim_orig, _ = text_similarity(first_500, second_500)
                pos_sim_orig, _ = pos_similarity(first_500, second_500)
                
                # Extract author (using the title as placeholder)
                author = title.replace("_", " ").title()
                
                # Determine model (placeholder - you may need to extract this from filename or content)
                model = "LLaMA"  # Placeholder
                
                # Create row for CSV
                row = {
                    "Author": author,
                    "Text_Length": original_length,
                    "Path_Full_text": original_path,
                    "First_500": first_500[:100] + "...",  # Truncated for CSV readability
                    "Second_500": second_500[:100] + "...", # Truncated for CSV readability
                    "Model": model,
                    "Path_generated_500": generated_path,
                    "Generated_500": generated_text[:500] + "...", # Truncated for CSV readability
                    "True_500": first_500[:100] + "...", # Truncated for CSV readability
                    "text_similarity_LLM": round(text_sim_llm, 4),
                    "text_similarity_ORIGINAL": round(text_sim_orig, 4),
                    "pos_similarity_LLM": round(pos_sim_llm, 4),
                    "pos_similarity_ORIGINAL": round(pos_sim_orig, 4),
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "comments": f"Comparison of {title}"
                }
                
                # Write row to CSV
                writer.writerow(row)
                
                print(f"✓ Processed essay: {title}")
                print(f"  - Text similarity (LLM): {text_sim_llm:.4f}")
                print(f"  - Text similarity (Original): {text_sim_orig:.4f}")
                print(f"  - POS similarity (LLM): {pos_sim_llm:.4f}")
                print(f"  - POS similarity (Original): {pos_sim_orig:.4f}")
                
            except Exception as e:
                print(f"Error processing essay {title}: {str(e)}")
    
    print(f"\nAnalysis complete. Results saved to {OUTPUT_FILE}")
    
    # Load and display summary statistics
    try:
        df = pd.read_csv(OUTPUT_FILE)
        print("\nSummary Statistics:")
        print("\nText Similarity Statistics:")
        print(df[["text_similarity_LLM", "text_similarity_ORIGINAL"]].describe())
        print("\nPOS Similarity Statistics:")
        print(df[["pos_similarity_LLM", "pos_similarity_ORIGINAL"]].describe())
    except Exception as e:
        print(f"Could not generate summary statistics: {str(e)}")

if __name__ == "__main__":
    analyze_essays()
