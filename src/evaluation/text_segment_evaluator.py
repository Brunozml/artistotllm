"""
Script to evaluate text by analyzing different segments.

example usage from command line:
    - Analyze a file
    python src/evaluation/text_segment_evaluator.py --file data/raw/some_text_file.txt

    - Analyze a string
    python src/evaluation/text_segment_evaluator.py --text "Your text to analyze here"

"""

import os
import sys
import argparse

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# example import of util
from src.utils import split_text

def analyze_segments(text):
    """
    Analyze different segments of text separately.
    
    Args:
        text (str): Input text to analyze
    """
    first_500, second_500, rest = split_text(text)
    
    print(f"First 500 words: {len(first_500.split())} words")
    print(f"Second 500 words: {len(second_500.split())} words")
    print(f"Rest of text: {len(rest.split())} words")
    
    # Here you could add more analysis of each segment
    # For example, calculating complexity, sentiment, etc.

def main():
    parser = argparse.ArgumentParser(description='Analyze text segments')
    parser.add_argument('--file', type=str, help='Path to the text file to analyze')
    parser.add_argument('--text', type=str, help='Text string to analyze')
    
    args = parser.parse_args()
    
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
        analyze_segments(text)
    elif args.text:
        analyze_segments(args.text)
    else:
        print("Please provide either a file path or text string to analyze")
        parser.print_help()

if __name__ == "__main__":
    main()
