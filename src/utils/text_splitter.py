"""
Utility functions for splitting texts into segments.
"""

def split_text(text):
    """
    Split a text into three parts: first 500 words, second 500 words, and the rest.
    
    Args:
        text (str): The input text to split
        
    Returns:
        tuple: (first_500_words, second_500_words, rest)
    """
    # Split the text into words
    words = text.split()
    
    # Get the three segments
    first_500 = ' '.join(words[:500]) if len(words) >= 500 else ' '.join(words)
    second_500 = ' '.join(words[500:1000]) if len(words) >= 1000 else ' '.join(words[500:])
    rest = ' '.join(words[1000:]) if len(words) > 1000 else ''
    
    return first_500, second_500, rest
