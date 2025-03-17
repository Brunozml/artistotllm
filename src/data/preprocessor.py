import re
from typing import List, Dict
import yaml

class TextPreprocessor:
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.chunk_size = self.config['data']['chunk_size']
        self.overlap = self.config['data']['overlap']
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += 1
            
            if current_length >= self.chunk_size:
                chunks.append(' '.join(current_chunk))
                # Keep last N words for overlap
                current_chunk = current_chunk[-self.overlap:]
                current_length = len(current_chunk)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def process_text(self, text: str) -> List[str]:
        """Process text through cleaning and chunking pipeline."""
        cleaned_text = self.clean_text(text)
        return self.chunk_text(cleaned_text)
    
    def process_file(self, file_path: str) -> List[str]:
        """Process a text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return self.process_text(text) 