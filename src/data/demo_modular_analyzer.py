#!/usr/bin/env python3
"""
Example script showing how to use the modular essay analyzer with custom metrics
"""

import sys
import os

# Add the project root to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Now we can import directly using relative imports
from analyze_essays_modular import EssayAnalyzer
from custom_metrics import register_metrics


def main():
    """Demo of using the modular essay analyzer with a custom metric"""
    # Set file paths
    original_dir = "data/PG_sample/processed"
    generated_dir = "data/PG_sample/generated"
    output_dir = "results/essay_analysis"
    
    print("Creating essay analyzer with custom metrics...")
    
    # Initialize the essay analyzer
    analyzer = EssayAnalyzer(
        original_dir=original_dir,
        generated_dir=generated_dir,
        output_dir=output_dir
    )
    
    # Register custom metrics
    print("Registering custom vocabulary diversity metric...")
    register_metrics(analyzer)
    
    # Analyze all essays
    print("Analyzing essays...")
    analyzer.analyze_all_essays()
    
    # Save results in both formats
    print("Saving results...")
    analyzer.save_results('both')
    
    print("\nAnalysis complete! Check the output files in the results directory.")


if __name__ == "__main__":
    main()
