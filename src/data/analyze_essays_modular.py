"""
Script to analyze and compare original essays with their generated counterparts using a modular metrics system.
This architecture makes it simple to add new metrics without modifying the core analysis logic.

Usage:
python analyze_essays_modular.py (defaults to CSV)
python analyze_essays_modular.py --format excel
python analyze_essays_modular.py --format both
"""

import os
import sys
import csv
import glob
import importlib
from datetime import datetime
import pandas as pd
import argparse
from typing import Callable, Dict, List, Any, Tuple

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.pos_similarity import pos_similarity, read_file
from evaluation.text_similarity import text_similarity
from evaluation.sentence_length import sentence_length_similarity
from evaluation.punctuation_similarity import punctuation_similarity
from evaluation.transformer_similarity import transformer_similarity, load_model
from utils.text_splitter import split_text

# Set file paths
ORIGINAL_DIR = "data/PG_sample/processed"
GENERATED_DIR = "data/PG_sample/generated"
OUTPUT_DIR = "results/essay_analysis"


class MetricProcessor:
    """
    A class to manage and process different metrics for essay comparison.
    Allows for easy addition of new metrics without modifying core logic.
    """
    
    def __init__(self):
        self.metrics = {}
        self.metric_columns = {}
    
    def register_metric(self, name: str, 
                        metric_function: Callable, 
                        output_columns: List[str] = None,
                        description: str = ""):
        """
        Register a new metric function with the processor
        
        Args:
            name: Unique identifier for the metric
            metric_function: Function that calculates the metric
            output_columns: List of column names this metric will produce
            description: Description of what the metric measures
        """
        self.metrics[name] = {
            "function": metric_function,
            "description": description
        }
        
        if output_columns:
            self.metric_columns[name] = output_columns
    
    def calculate_metrics(self, first_text: str, second_text: str, compare_text: str = None) -> Dict:
        """
        Calculate all registered metrics for the given texts
        
        Args:
            first_text: First text to compare (typically first 500 words of original)
            second_text: Second text to compare (typically generated text)
            compare_text: Optional third text for comparison (typically second 500 words of original)
            
        Returns:
            Dictionary containing all metric results
        """
        results = {}
        
        for metric_name, metric_info in self.metrics.items():
            try:
                # Calculate metric between first text and second text (LLM comparison)
                llm_result, llm_details = metric_info["function"](first_text, second_text)
                
                # Add main metric result
                results[f"{metric_name}_LLM"] = round(llm_result, 4)
                
                # Add any additional details from the metric 
                if isinstance(llm_details, dict):
                    for key, value in llm_details.items():
                        if isinstance(value, dict) and "avg_sentence_length" in value:
                            results[f"avg_{metric_name}_LLM"] = round(value["avg_sentence_length"], 2)
                
                # If we have a third text for comparison (original text continued)
                if compare_text:
                    orig_result, orig_details = metric_info["function"](first_text, compare_text)
                    results[f"{metric_name}_ORIGINAL"] = round(orig_result, 4)
                    
                    if isinstance(orig_details, dict):
                        for key, value in orig_details.items():
                            if isinstance(value, dict) and "avg_sentence_length" in value:
                                results[f"avg_{metric_name}_ORIGINAL"] = round(value["avg_sentence_length"], 2)
            
            except Exception as e:
                print(f"Error calculating {metric_name}: {str(e)}")
                results[f"{metric_name}_LLM"] = None
                if compare_text:
                    results[f"{metric_name}_ORIGINAL"] = None
        
        return results


class EssayAnalyzer:
    """
    Main class to analyze essays using the registered metrics
    """
    
    def __init__(self, original_dir: str, generated_dir: str, output_dir: str):
        self.original_dir = original_dir
        self.generated_dir = generated_dir
        self.output_dir = output_dir
        self.metric_processor = MetricProcessor()
        self.results = []
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Register default metrics
        self._register_default_metrics()
        
        # Load transformer model once for all essay comparisons
        self.transformer_model = load_model()
    
    def _register_default_metrics(self):
        """Register the default set of metrics"""
        self.metric_processor.register_metric(
            "text_similarity", 
            text_similarity,
            description="Measures text similarity between essays"
        )
        
        self.metric_processor.register_metric(
            "pos_similarity", 
            pos_similarity,
            description="Measures part-of-speech tag similarity between essays"
        )
        
        self.metric_processor.register_metric(
            "sentence_similarity", 
            sentence_length_similarity,
            description="Measures sentence length similarity between essays"
        )
        
        self.metric_processor.register_metric(
            "punctuation_similarity", 
            punctuation_similarity,
            description="Measures punctuation pattern similarity between essays"
        )
        
        # Add transformer similarity metric
        self.metric_processor.register_metric(
            "transformer_similarity",
            lambda text1, text2: transformer_similarity(text1, text2, self.transformer_model),
            description="Measures semantic similarity using transformer embeddings"
        )

    def register_new_metric(self, name, metric_function, output_columns=None, description=""):
        """
        Register a new metric to use in analysis
        
        Args:
            name: Name of the metric
            metric_function: Function to calculate the metric
            output_columns: Column names this metric produces
            description: Description of the metric
        """
        self.metric_processor.register_metric(name, metric_function, output_columns, description)
    
    def get_essay_pairs(self) -> List[Tuple[str, str, str]]:
        """Find all essay pairs (original and generated versions)"""
        pairs = []
        
        # Get all original essay files
        original_files = glob.glob(os.path.join(self.original_dir, "*.txt"))
        
        for original_file in original_files:
            title = os.path.basename(original_file).replace('.txt', '')
            generated_file = os.path.join(self.generated_dir, f"{title}_generated.txt")
            
            # Check if the generated file exists
            if os.path.exists(generated_file):
                pairs.append((title, original_file, generated_file))
        
        return pairs
    
    def analyze_essay_pair(self, title: str, original_path: str, generated_path: str) -> Dict:
        """
        Analyze a single essay pair
        
        Args:
            title: Essay title
            original_path: Path to original essay
            generated_path: Path to generated essay
            
        Returns:
            Dictionary with analysis results
        """
        # Read the original and generated texts
        original_text = read_file(original_path)
        generated_text = read_file(generated_path)
        
        # Split the original text into segments
        first_500, second_500, rest = split_text(original_text)
        
        # Calculate text lengths
        original_length = len(original_text.split())
        generated_length = len(generated_text.split())
        
        # Extract author (using the title as placeholder)
        author = title.replace("_", " ").title()
        
        # Determine model (placeholder - you may need to extract this from filename or content)
        model = "Llama-3.2-1B-Instruct"  # Placeholder
        
        # Create base result with metadata
        result = {
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
            "date": datetime.now().strftime("%d-%m-%Y"),
            "comments": f"Comparison of {title}"
        }
        
        # Calculate all metrics for this essay pair
        metrics_result = self.metric_processor.calculate_metrics(
            first_text=first_500,
            second_text=generated_text,
            compare_text=second_500
        )
        
        # Merge metrics with base result
        result.update(metrics_result)
        
        return result
    
    def analyze_all_essays(self) -> List[Dict]:
        """
        Analyze all essay pairs in the directories
        
        Returns:
            List of dictionaries with analysis results
        """
        # Get all essay pairs
        essay_pairs = self.get_essay_pairs()
        print(f"Found {len(essay_pairs)} essay pairs to analyze")
        
        # Process each pair
        results = []
        for title, original_path, generated_path in essay_pairs:
            print(f"\nAnalyzing essay: {title}")
            
            try:
                # Analyze this essay pair
                result = self.analyze_essay_pair(title, original_path, generated_path)
                results.append(result)
                
                # Print some results
                for key, value in result.items():
                    if "similarity" in key.lower() and isinstance(value, (int, float)):
                        print(f"  - {key}: {value:.4f}")
                
                print(f"✓ Processed essay: {title}")
                
            except Exception as e:
                print(f"Error processing essay {title}: {str(e)}")
        
        self.results = results
        return results
    
    def save_results(self, output_format='csv') -> Tuple[str, str]:
        """
        Save analysis results to files
        
        Args:
            output_format: 'csv', 'excel', or 'both'
            
        Returns:
            Tuple of (csv_path, excel_path) where applicable
        """
        if not self.results:
            print("No results to save")
            return None, None
        
        # Generate output filenames
        timestamp = datetime.now().strftime('%d%m%Y_%H%M')
        output_csv = os.path.join(self.output_dir, f"essay_comparison_{timestamp}.csv")
        output_excel = os.path.join(self.output_dir, f"essay_comparison_{timestamp}.xlsx")
        
        # Save results based on the requested output format
        if output_format in ['csv', 'both']:
            # Save as CSV
            fieldnames = list(self.results[0].keys())
            with open(output_csv, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)
            print(f"\nCSV results saved to {output_csv}")
        else:
            output_csv = None
        
        if output_format in ['excel', 'both']:
            # Save as Excel
            df = pd.DataFrame(self.results)
            
            # Create an Excel writer
            with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
                # Write the main data sheet
                df.to_excel(writer, sheet_name='Analysis Results', index=False)
                
                # Create a summary sheet with statistics
                # Find all metric columns (those containing 'similarity')
                metric_cols = [col for col in df.columns if 'similarity' in col.lower()]
                avg_cols = [col for col in df.columns if col.startswith('avg_') and col.endswith(('_LLM', '_ORIGINAL'))]
                
                # Create summary statistics
                summary_data = {'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max']}
                
                # Add data for each metric column
                for col in metric_cols + avg_cols:
                    if col in df.columns:
                        summary_data[col] = [
                            df[col].mean(),
                            df[col].median(),
                            df[col].std(),
                            df[col].min(),
                            df[col].max()
                        ]
                
                # Create the summary DataFrame
                summary = pd.DataFrame(summary_data)
                summary.to_excel(writer, sheet_name='Summary Statistics', index=False)
                
                # Format the summary sheet
                workbook = writer.book
                worksheet = writer.sheets['Summary Statistics']
                
                # Add number format
                number_format = workbook.add_format({'num_format': '0.0000'})
                for col in range(1, len(summary_data)):
                    worksheet.set_column(col, col, 12, number_format)
                
            print(f"\nExcel results saved to {output_excel}")
        else:
            output_excel = None
        
        # Display summary statistics
        self._print_summary_statistics()
        
        return output_csv, output_excel
    
    def _print_summary_statistics(self):
        """Print summary statistics to console"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        print("\nSummary Statistics:")
        
        # Group metrics by type
        metric_groups = {}
        
        for col in df.columns:
            if 'similarity' in col.lower():
                metric_type = col.split('_')[0]
                if metric_type not in metric_groups:
                    metric_groups[metric_type] = []
                metric_groups[metric_type].append(col)
        
        # Print summary for each metric group
        for metric_type, columns in metric_groups.items():
            print(f"\n{metric_type.title()} Similarity Statistics:")
            print(df[columns].describe())


def main():
    """Main function to run the essay analysis"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze and compare essays')
    parser.add_argument('--format', type=str, default='csv', 
                        choices=['csv', 'excel', 'both'],
                        help='Output format: csv, excel, or both (default: csv)')
    
    # Add an argument for importing custom metrics
    parser.add_argument('--custom-metrics', type=str, nargs='+',
                        help='Python modules containing custom metrics to import (e.g., my_metrics)')
    
    args = parser.parse_args()
    
    # Initialize the essay analyzer
    analyzer = EssayAnalyzer(
        original_dir=ORIGINAL_DIR,
        generated_dir=GENERATED_DIR,
        output_dir=OUTPUT_DIR
    )
    
    # Import custom metrics if specified
    if args.custom_metrics:
        for module_name in args.custom_metrics:
            try:
                # Import the module dynamically
                module = importlib.import_module(module_name)
                
                # Look for a register_metrics function in the module
                if hasattr(module, 'register_metrics'):
                    module.register_metrics(analyzer)
                    print(f"Registered metrics from {module_name}")
                else:
                    print(f"Warning: Module {module_name} does not have a register_metrics function")
            except ImportError as e:
                print(f"Error importing module {module_name}: {str(e)}")
    
    # Analyze all essays
    analyzer.analyze_all_essays()
    
    # Save results in the specified format
    analyzer.save_results(args.format)


if __name__ == "__main__":
    main()
