import os
from preprocessor import TextPreprocessor
import json
from sklearn.model_selection import train_test_split

def process_ethics():
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Process the text
    chunks = preprocessor.process_file('data/raw/ethics.txt.utf-8')
    
    # Split into train/val/test
    train_chunks, temp_chunks = train_test_split(
        chunks, 
        test_size=0.2, 
        random_state=42
    )
    val_chunks, test_chunks = train_test_split(
        temp_chunks, 
        test_size=0.5, 
        random_state=42
    )
    
    # Create output directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Save processed chunks
    for split_name, split_chunks in [
        ('train', train_chunks),
        ('val', val_chunks),
        ('test', test_chunks)
    ]:
        output_file = f'data/processed/ethics_{split_name}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_chunks, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(split_chunks)} chunks to {output_file}")

if __name__ == "__main__":
    process_ethics() 