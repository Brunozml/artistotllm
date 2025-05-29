"""
🤖 Transformer Text Telepathy! 🧠

Behold the power of neural networks! This module uses state-of-the-art transformer models
to read between the lines and understand text like a mind reader. It transforms your words
into high-dimensional vectors and measures their cosmic similarity in the neural space.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Tuple

def load_model():
    """Load the Sentence-BERT model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def transformer_similarity(text1: str, text2: str, model=None) -> Tuple[float, dict]:
    """
    Compare two texts using Sentence Transformers.
    Returns a similarity score and the sentence embeddings.
    """
    if model is None:
        model = load_model()
    
    # Get embeddings
    embedding1 = model.encode([text1])[0]
    embedding2 = model.encode([text2])[0]
    
    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    return similarity, {
        "embedding1_sample": embedding1[:5].tolist(),  # Show first 5 dimensions
        "embedding2_sample": embedding2[:5].tolist()
    }

def read_file(filepath):
    """Read text from a file and return its contents"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

if __name__ == "__main__":
    # Read the files
    file1 = 'data/raw/pg_essays/what_to_do.txt'	
    file2 = 'data/raw/pg_llama/what_to_do.txt'

    text1 = read_file(file1)
    text2 = read_file(file2)
    
    # Load model once
    model = load_model()
    
    # Compare texts
    similarity_score, embeddings = transformer_similarity(text1, text2, model)
    
    # Print results
    print(f"Text 1: {file1}")
    print(f"Text 2: {file2}")
    print(f"\nTransformer Similarity score: {similarity_score:.2f}")
    print("\nSample of embeddings (first 5 dimensions):")
    print(f"Text 1: {embeddings['embedding1_sample']}")
    print(f"Text 2: {embeddings['embedding2_sample']}")
