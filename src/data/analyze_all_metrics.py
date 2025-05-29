#!/usr/bin/env python3
"""
Script to analyze text metrics for results_data.csv.

This script loads data/processed/results_data.csv and applies all metrics
from src/evaluation to compare:
1) y_train vs y_test (baseline)
2) y_train vs y_pred (model prediction)

Output is saved as CSV or Excel file with summary statistics.

Usage:
    python analyze_all_metrics.py (defaults to CSV)
    python analyze_all_metrics.py --format excel
    python analyze_all_metrics.py --format both
"""

import os
import sys
import csv
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all evaluation metrics
from evaluation.text_similarity import text_similarity
from evaluation.pos_similarity import pos_similarity
from evaluation.sentence_length import sentence_length_similarity
from evaluation.tfidf_similarity import tfidf_similarity
from evaluation.transformer_similarity import transformer_similarity
from evaluation.stopwords_similarity import stopwords_similarity
from evaluation.punctuation_similarity import punctuation_similarity
from evaluation.ttr_similarity import ttr_similarity

# Set file paths
INPUT_CSV = "data/processed/final_dataset.csv"
OUTPUT_DIR = "results/metrics_analysis"

def analyze_metrics(output_format='csv'):
    """
    Analyze text using all available metrics
    
    Args:
        output_format (str): Format to save results - 'csv', 'excel', or 'both'
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate output filenames
    timestamp = datetime.now().strftime('%d%B%Y_%H%M')
    output_csv = os.path.join(OUTPUT_DIR, f"metrics_analysis_{timestamp}.csv")
    output_excel = os.path.join(OUTPUT_DIR, f"metrics_analysis_{timestamp}.xlsx")
    
    # Check if input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file {INPUT_CSV} not found!")
        return
        
    # Read the CSV file
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"Successfully loaded {INPUT_CSV} with {len(df)} rows")
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        return
        
    # Initialize results list
    results = []
    
    # Define metrics list with their names, functions, and any special parameters
    metrics = [
        {"name": "Text Similarity", "func": text_similarity, "params": {}},
        {"name": "POS Similarity", "func": pos_similarity, "params": {}},
        {"name": "Sentence Length Similarity", "func": sentence_length_similarity, "params": {}},
        {"name": "TF-IDF Similarity Unigram", "func": tfidf_similarity, "params": {"ngram_range": (1, 1), "remove_stopwords": False}},
        {"name": "TF-IDF Similarity Bigram", "func": tfidf_similarity, "params": {"ngram_range": (2, 2), "remove_stopwords": False}},
        {"name": "TF-IDF Similarity Trigram", "func": tfidf_similarity, "params": {"ngram_range": (3, 3), "remove_stopwords": False}},
        {"name": "Transformer Similarity", "func": transformer_similarity, "params": {}},
        {"name": "Stopwords Similarity", "func": stopwords_similarity, "params": {}},
        {"name": "Punctuation Similarity", "func": punctuation_similarity, "params": {}},
        {"name": "TTR Similarity", "func": ttr_similarity, "params": {"include_stopwords": True}}
    ]
    
    # Load transformer model once to reuse for efficiency
    transformer_model = None
    try:
        from evaluation.transformer_similarity import load_model
        transformer_model = load_model()
        print("Loaded transformer model for similarity calculations")
    except Exception as e:
        print(f"Could not load transformer model: {str(e)}")
        print("Will load model on demand (may be slower)")
    
    # Process each row in the dataset
    for index, row in df.iterrows():
        print(f"\nAnalyzing entry {index+1}/{len(df)}: {row['Author']} - {row['Title']}")
        
        try:
            # Extract texts
            y_train = row['y_train']
            y_pred = row['y_pred']
            y_test = row['y_test']
            
            # Skip if any of the texts is empty
            if pd.isna(y_train) or pd.isna(y_pred) or pd.isna(y_test):
                print(f"  Skipping due to missing data in entry")
                continue
                
            # Initialize row result
            result = {
                "Author": row['Author'],
                "Title": row['Title'],
                "y_train": y_train,
                "y_test": y_test,
                "y_pred": y_pred,
                "train_length": len(y_train.split()),
                "test_length": len(y_test.split()),
                "pred_length": len(y_pred.split())
            }
            
            # Apply each metric twice
            for metric in metrics:
                metric_name = metric["name"]
                metric_func = metric["func"]
                print(f"  Calculating {metric_name}...")
                
                try:
                    params = metric["params"]
                    
                    # Handle special case for transformer similarity
                    if metric_name == "Transformer Similarity" and transformer_model is not None:
                        params = {"model": transformer_model}
                    
                    # Special handling for punctuation similarity to prevent NaN values
                    if metric_name == "Punctuation Similarity":
                        try:
                            # Apply to y_train vs y_test (baseline)
                            baseline_score, baseline_details = metric_func(y_train, y_test, **params)
                            
                            # Apply to y_train vs y_pred (prediction)
                            pred_score, pred_details = metric_func(y_train, y_pred, **params)
                            
                            if np.isnan(baseline_score) or np.isnan(pred_score):
                                raise ValueError("Punctuation similarity returned NaN values")
                                
                            # Add to result
                            metric_key = metric_name.replace(" ", "_")
                            result[f"{metric_key}_baseline"] = round(baseline_score, 4)
                            result[f"{metric_key}_prediction"] = round(pred_score, 4)
                            
                        except Exception as punct_error:
                            print(f"    Error in punctuation similarity calculation: {str(punct_error)}")
                            result[f"Punctuation_Similarity_baseline"] = None
                            result[f"Punctuation_Similarity_prediction"] = None
                    else:
                        # Special handling for stopwords similarity to prevent type errors
                        if metric_name == "Stopwords Similarity":
                            try:
                                # Apply to y_train vs y_test (baseline)
                                baseline_score, baseline_details = metric_func(y_train, y_test, **params)
                                
                                # Apply to y_train vs y_pred (prediction)
                                pred_score, pred_details = metric_func(y_train, y_pred, **params)
                                
                                if np.isnan(baseline_score) or np.isnan(pred_score):
                                    raise ValueError("Stopwords similarity returned NaN values")
                                    
                                # Add to result
                                metric_key = metric_name.replace(" ", "_")
                                result[f"{metric_key}_baseline"] = round(baseline_score, 4)
                                result[f"{metric_key}_prediction"] = round(pred_score, 4)
                                
                            except Exception as sw_error:
                                print(f"    Error in stopwords similarity calculation: {str(sw_error)}")
                                result[f"Stopwords_Similarity_baseline"] = None
                                result[f"Stopwords_Similarity_prediction"] = None
                        else:
                            # Standard processing for all other metrics
                            baseline_score, baseline_details = metric_func(y_train, y_test, **params)
                            pred_score, pred_details = metric_func(y_train, y_pred, **params)
                            
                            # Add to result
                            metric_key = metric_name.replace(" ", "_")
                            result[f"{metric_key}_baseline"] = round(baseline_score, 4)
                            result[f"{metric_key}_prediction"] = round(pred_score, 4)
                            
                except Exception as e:
                    print(f"    Error calculating {metric_name}: {str(e)}")
                    result[f"{metric_key}_baseline"] = None
                    result[f"{metric_key}_prediction"] = None
            
            # Add metadata
            result["date"] = datetime.now().strftime("%d-%m-%Y")
            
            # Add to results list
            results.append(result)
            print(f"✓ Processed entry: {row['Author']} - {row['Title']}")
            
        except Exception as e:
            print(f"Error processing row {index}: {str(e)}")
    
    # Convert results to DataFrame
    if not results:
        print("Error: No results generated!")
        return
        
    df_results = pd.DataFrame(results)
    
    # Save results based on the requested output format
    if output_format in ['csv', 'both']:
        # Save as CSV
        df_results.to_csv(output_csv, index=False)
        print(f"\nCSV results saved to {output_csv}")
    
    if output_format in ['excel', 'both']:
        # Save as Excel
        with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
            # Write the main data sheet
            df_results.to_excel(writer, sheet_name='Analysis Results', index=False)
            
            # Create a summary sheet with statistics
            # Get metric columns
            metric_cols = [col for col in df_results.columns 
                          if any(m["name"].replace(" ", "_") in col for m in metrics)]
            
            # Initialize summary data dictionary with metrics and statistics
            summary_data = {'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max']}
            
            # Add data for each metric column
            for col in metric_cols:
                if df_results[col].dtype in [float, int]:
                    summary_data[col] = [
                        df_results[col].mean(),
                        df_results[col].median(),
                        df_results[col].std(),
                        df_results[col].min(),
                        df_results[col].max()
                    ]
            
            # Create summary dataframe and write to Excel
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
            
            # Format the summary sheet
            workbook = writer.book
            worksheet = writer.sheets['Summary Statistics']
            
            # Add number format for numeric columns
            number_format = workbook.add_format({'num_format': '0.0000'})
            for col in range(1, len(summary_data.keys())):
                worksheet.set_column(col, col, 12, number_format)
                
            # Create author-specific summary sheets
            authors = df_results['Author'].unique()
            print(f"\nCreating summary sheets for {len(authors)} authors...")
            
            for author in authors:
                # Filter data for this author
                author_df = df_results[df_results['Author'] == author]
                
                # Skip if author has very few entries
                if len(author_df) < 3:
                    print(f"  Skipping {author}: too few entries ({len(author_df)})")
                    continue
                    
                # Create a valid sheet name (Excel limits sheet names to 31 chars)
                sheet_name = f"{author[:25]}" if len(author) > 25 else author
                sheet_name = sheet_name.replace('/', '_').replace('\\', '_')
                print(f"  Creating summary sheet for {author} ({len(author_df)} entries)")
                
                # Initialize author summary data
                author_summary = {'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Count']}
                
                # Add data for each metric column
                for col in metric_cols:
                    if author_df[col].dtype in [float, int]:
                        author_summary[col] = [
                            author_df[col].mean(),
                            author_df[col].median(),
                            author_df[col].std(),
                            author_df[col].min(),
                            author_df[col].max(),
                            author_df[col].count()
                        ]
                
                # Create author summary dataframe and write to Excel
                author_summary_df = pd.DataFrame(author_summary)
                author_summary_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Format the author sheet
                worksheet = writer.sheets[sheet_name]
                for col in range(1, len(author_summary.keys())):
                    worksheet.set_column(col, col, 12, number_format)
                
        print(f"\nExcel results saved to {output_excel}")
    
    # Display summary statistics
    print("\nSummary Statistics:")
    
    # Group metrics for display
    metric_groups = {}
    for metric in metrics:
        metric_key = metric["name"].replace(" ", "_")
        baseline_col = f"{metric_key}_baseline"
        prediction_col = f"{metric_key}_prediction"
        
        if baseline_col in df_results.columns and prediction_col in df_results.columns:
            print(f"\n{metric['name']} Statistics:")
            stat_cols = [baseline_col, prediction_col]
            print(df_results[stat_cols].describe())

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze text using all metrics')
    parser.add_argument('--format', type=str, default='csv', 
                        choices=['csv', 'excel', 'both'],
                        help='Output format: csv, excel, or both (default: csv)')
    args = parser.parse_args()
    
    # Run analysis with specified output format
    analyze_metrics(args.format)
