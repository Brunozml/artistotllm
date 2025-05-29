#!/usr/bin/env python3
"""
Script to process the results.csv file.
This script loads the CSV file, keeps only selected columns
(Author, Title, y_train, y_pred, y_test), and saves it as results_data.csv.
"""

import pandas as pd
import os
from pathlib import Path

def process_csv():
    # Define the paths to the csv files
    project_root = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    input_csv_path = project_root / "data" / "processed" / "results.csv"
    output_csv_path = project_root / "data" / "processed" / "results_data.csv"
    
    # Check if the input file exists
    if not input_csv_path.exists():
        print(f"Error: File not found at {input_csv_path}")
        return
    
    # Load the CSV file
    try:
        df = pd.read_csv(input_csv_path)
         # Keep only the specified columns
        columns_to_keep = ['Author', 'Title', 'y_train', 'y_pred', 'y_test']
        df_filtered = df[columns_to_keep]
        
        # Save the filtered dataframe to a new CSV file
        df_filtered.to_csv(output_csv_path, index=False)
        
        # Print summary information
        print(f"Successfully created {output_csv_path} with only the specified columns:")
        for col in columns_to_keep:
            print(f"- {col}")
        
        print(f"\nNumber of rows: {len(df_filtered)}")
        print(f"Number of columns: {len(df_filtered.columns)}")
            
    except Exception as e:
        print(f"Error processing CSV file: {e}")

if __name__ == "__main__":
    process_csv()
