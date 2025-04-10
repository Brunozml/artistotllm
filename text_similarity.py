def read_file(filepath):
    """Read text from a file and return its contents"""
    with open(filepath, 'r') as f:
        return f.read()

def compare_texts(text1, text2):
    """
    Compare similarity between two texts using a simple word overlap approach.
    Returns a similarity score between 0 and 1.
    """
    # Convert to lowercase and split into words, removing punctuation
    import re
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    
    # Find common words
    common_words = words1.intersection(words2)
    
    # Calculate similarity score
    similarity = len(common_words) / max(len(words1), len(words2))
    
    return similarity, common_words

# Read the files
file1 = '/Users/brunozorrilla/Documents/GitHub/artistotllm/data/raw/hypewrite_what_to_do.txt'
file2 = '/Users/brunozorrilla/Documents/GitHub/artistotllm/data/raw/gpt_what_to_do.txt'

text1 = read_file(file1)
text2 = read_file(file2)

# Compare the texts
similarity_score, shared_words = compare_texts(text1, text2)

# Print results
print(f"Comparing:\n1. {file1}\n2. {file2}\n")
print(f"Similarity score: {similarity_score:.2f}")
print(f"Number of shared words: {len(shared_words)}")
print(f"Sample of shared words (up to 10): {list(shared_words)[:10]}")