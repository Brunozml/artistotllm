"""
Script to analyze and compare original essays with their generated counterparts.
This script calculates text and POS similarity metrics and stores results in a CSV file.

Now users can run the script with options:

python analyze_essays.py (defaults to CSV)
python analyze_essays.py --format excel
python analyze_essays.py --format both
"""

import os
import sys
import csv
import glob
from datetime import datetime
import pandas as pd
import argparse

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.pos_similarity import pos_similarity, read_file
from evaluation.text_similarity import text_similarity
from evaluation.sentence_length import sentence_length_similarity
from utils.text_splitter import split_text

# Set file paths
ORIGINAL_DIR = "data/PG_sample/processed"
GENERATED_DIR = "data/PG_sample/generated"
OUTPUT_DIR = "results/essay_analysis"

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

def analyze_essays(output_format='csv'):
    """
    Analyze and compare all essay pairs
    
    Args:
        output_format (str): Format to save results - 'csv', 'excel', or 'both'
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate output filenames
    timestamp = datetime.now().strftime('%d%m%Y_%H%M')
    output_csv = os.path.join(OUTPUT_DIR, f"essay_comparison_{timestamp}.csv")
    output_excel = os.path.join(OUTPUT_DIR, f"essay_comparison_{timestamp}.xlsx")
    
    # Get all essay pairs
    essay_pairs = get_essay_pairs()
    print(f"Found {len(essay_pairs)} essay pairs to analyze")
    
    # Initialize results list to store all data
    results = []
    
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
            
            # Calculate text similarity metrics
            # 1. First 500 words of original vs generated
            text_sim_llm, _ = text_similarity(first_500, generated_text)
            pos_sim_llm, _ = pos_similarity(first_500, generated_text)
            sent_sim_llm, sent_stats_llm = sentence_length_similarity(first_500, generated_text)
            
            # 2. First 500 words vs second 500 words of original
            text_sim_orig, _ = text_similarity(first_500, second_500)
            pos_sim_orig, _ = pos_similarity(first_500, second_500)
            sent_sim_orig, sent_stats_orig = sentence_length_similarity(first_500, second_500)
            
            # Extract author (using the title as placeholder)
            author = title.replace("_", " ").title()
            
            # Determine model (placeholder - you may need to extract this from filename or content)
            model = "Llama-3.2-1B-Instruct"  # Placeholder
            
            # Create row for results
            row = {
                "Author": author,
                "Title": title,
                "Text_Length": original_length,
                "Path_Full_text": original_path,
                "First_500": first_500 + "...",  
                "Second_500": "..." + second_500 + "...", 
                "Model": model,
                "Path_generated_500": generated_path,
                "Generated_500": "..." + generated_text,
                "generated_length": generated_length,
                "text_similarity_LLM": round(text_sim_llm, 4),
                "text_similarity_ORIGINAL": round(text_sim_orig, 4),
                "pos_similarity_LLM": round(pos_sim_llm, 4),
                "pos_similarity_ORIGINAL": round(pos_sim_orig, 4),
                "sentence_similarity_LLM": round(sent_sim_llm, 4),
                "sentence_similarity_ORIGINAL": round(sent_sim_orig, 4),
                "avg_sentence_length_ORIGINAL": round(sent_stats_orig["text1_stats"]["avg_sentence_length"], 2),
                "avg_sentence_length_LLM": round(sent_stats_llm["text2_stats"]["avg_sentence_length"], 2),
                "sentence_length_diff": round(sent_stats_llm["difference"], 2),
                "date": datetime.now().strftime("%d-%m-%Y"),
                "comments": f"Comparison of {title}"
            }
            
            # Add row to results
            results.append(row)
            
            print(f"✓ Processed essay: {title}")
            print(f"  - Text similarity (LLM): {text_sim_llm:.4f}")
            print(f"  - Text similarity (Original): {text_sim_orig:.4f}")
            print(f"  - POS similarity (LLM): {pos_sim_llm:.4f}")
            print(f"  - POS similarity (Original): {pos_sim_orig:.4f}")
            print(f"  - Sentence Length similarity (LLM): {sent_sim_llm:.4f}")
            print(f"  - Avg sentence length (Original): {sent_stats_orig['text1_stats']['avg_sentence_length']:.2f}")
            print(f"  - Avg sentence length (LLM): {sent_stats_llm['text2_stats']['avg_sentence_length']:.2f}")
            
        except Exception as e:
            print(f"Error processing essay {title}: {str(e)}")
    
    # Save results based on the requested output format
    if output_format in ['csv', 'both']:
        # Save as CSV
        fieldnames = list(results[0].keys()) if results else []
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV results saved to {output_csv}")
    
    if output_format in ['excel', 'both']:
        # Save as Excel
        df = pd.DataFrame(results)
        
        # Create an Excel writer
        with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
            # Write the main data sheet
            df.to_excel(writer, sheet_name='Analysis Results', index=False)
            
            # Create a summary sheet with statistics
            summary = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Text Sim (LLM)': [
                    df['text_similarity_LLM'].mean(),
                    df['text_similarity_LLM'].median(),
                    df['text_similarity_LLM'].std(),
                    df['text_similarity_LLM'].min(),
                    df['text_similarity_LLM'].max()
                ],
                'Text Sim (Original)': [
                    df['text_similarity_ORIGINAL'].mean(),
                    df['text_similarity_ORIGINAL'].median(),
                    df['text_similarity_ORIGINAL'].std(),
                    df['text_similarity_ORIGINAL'].min(),
                    df['text_similarity_ORIGINAL'].max()
                ],
                'POS Sim (LLM)': [
                    df['pos_similarity_LLM'].mean(),
                    df['pos_similarity_LLM'].median(),
                    df['pos_similarity_LLM'].std(),
                    df['pos_similarity_LLM'].min(),
                    df['pos_similarity_LLM'].max()
                ],
                'POS Sim (Original)': [
                    df['pos_similarity_ORIGINAL'].mean(),
                    df['pos_similarity_ORIGINAL'].median(),
                    df['pos_similarity_ORIGINAL'].std(),
                    df['pos_similarity_ORIGINAL'].min(),
                    df['pos_similarity_ORIGINAL'].max()
                ],
                'Sent Len Sim (LLM)': [
                    df['sentence_similarity_LLM'].mean(),
                    df['sentence_similarity_LLM'].median(),
                    df['sentence_similarity_LLM'].std(),
                    df['sentence_similarity_LLM'].min(),
                    df['sentence_similarity_LLM'].max()
                ],
                'Sent Len Sim (Original)': [
                    df['sentence_similarity_ORIGINAL'].mean(),
                    df['sentence_similarity_ORIGINAL'].median(),
                    df['sentence_similarity_ORIGINAL'].std(),
                    df['sentence_similarity_ORIGINAL'].min(),
                    df['sentence_similarity_ORIGINAL'].max()
                ],
                'Avg Sent Length (Original)': [
                    df['avg_sentence_length_ORIGINAL'].mean(),
                    df['avg_sentence_length_ORIGINAL'].median(),
                    df['avg_sentence_length_ORIGINAL'].std(),
                    df['avg_sentence_length_ORIGINAL'].min(),
                    df['avg_sentence_length_ORIGINAL'].max()
                ],
                'Avg Sent Length (LLM)': [
                    df['avg_sentence_length_LLM'].mean(),
                    df['avg_sentence_length_LLM'].median(),
                    df['avg_sentence_length_LLM'].std(),
                    df['avg_sentence_length_LLM'].min(),
                    df['avg_sentence_length_LLM'].max()
                ]
            })
            
            summary.to_excel(writer, sheet_name='Summary Statistics', index=False)
            
            # Format the summary sheet
            workbook = writer.book
            worksheet = writer.sheets['Summary Statistics']
            
            # Add number format
            number_format = workbook.add_format({'num_format': '0.0000'})
            for col in range(1, 9):
                worksheet.set_column(col, col, 12, number_format)
            
        print(f"\nExcel results saved to {output_excel}")
    
    # Display summary statistics
    if results:
        df = pd.DataFrame(results)
        print("\nSummary Statistics:")
        
        print("\nText Similarity Statistics:")
        print(df[["text_similarity_LLM", "text_similarity_ORIGINAL"]].describe())
        
        print("\nPOS Similarity Statistics:")
        print(df[["pos_similarity_LLM", "pos_similarity_ORIGINAL"]].describe())
        
        print("\nSentence Length Statistics:")
        print(df[["sentence_similarity_LLM", "sentence_similarity_ORIGINAL", 
                 "avg_sentence_length_ORIGINAL", "avg_sentence_length_LLM"]].describe())

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze and compare essays')
    parser.add_argument('--format', type=str, default='csv', 
                        choices=['csv', 'excel', 'both'],
                        help='Output format: csv, excel, or both (default: csv)')
    args = parser.parse_args()
    
    # Run analysis with specified output format
    analyze_essays(args.format)
