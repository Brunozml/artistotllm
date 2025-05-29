import os
import csv
import glob
import chardet
import shutil

# Directory containing the essays
essays_dir = 'data/raw/Paul Krugman' 
output_file = 'data/long_krugman_essay_lengths.csv'
long_essays_dir = 'data/krugman_hansen_essays'  # Directory for essays with more than 1k words

def count_words(text):
    """Count the number of words in a text."""
    return len(text.split())

def detect_encoding(file_path):
    """Detect the encoding of a file."""
    with open(file_path, 'rb') as file:
        # Read a sample of the file to detect encoding
        raw_data = file.read(min(32768, os.path.getsize(file_path)))
    result = chardet.detect(raw_data)
    return result['encoding']

def read_file_with_fallback(file_path):
    """Read a file with encoding detection and fallbacks."""
    # Try to detect encoding first
    try:
        detected_encoding = detect_encoding(file_path)
        if detected_encoding:
            with open(file_path, 'r', encoding=detected_encoding) as file:
                return file.read()
    except Exception:
        pass
    
    # If detection fails or reading with detected encoding fails, try these encodings
    encodings = ['utf-8', 'latin-1', 'utf-16', 'ascii', 'windows-1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except Exception:
            continue
    
    # If all else fails, use latin-1 which can read any byte sequence
    with open(file_path, 'r', encoding='latin-1') as file:
        return file.read()

def process_essays():
    """Process all .txt files and store their word counts in a CSV.
    Also count and print the number of essays with more than 1k words.
    Copy essays with more than 1k words to a separate directory.
    """
    # Make sure the output directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(long_essays_dir, exist_ok=True)
    
    # Get all .txt files in the directory
    txt_files = glob.glob(os.path.join(essays_dir, '*.txt'))
    
    # Create a list to store results
    results = []
    essays_over_1k = 0  # Counter for essays with more than 1k words
    
    # Process each file
    for file_path in txt_files:
        filename = os.path.basename(file_path)
        
        try:
            content = read_file_with_fallback(file_path)
            word_count = count_words(content)
            results.append((filename, word_count))
            
            if word_count > 1000:
                essays_over_1k += 1
                # Copy the file to the long essays directory
                dest_path = os.path.join(long_essays_dir, filename)
                shutil.copy2(file_path, dest_path)
                print(f"Copied {filename} to {long_essays_dir} ({word_count} words)")
            else:
                print(f"Processed {filename}: {word_count} words")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Write results to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Filename', 'Word Count'])
        csv_writer.writerows(results)
    
    print(f"Results saved to {output_file}")
    print(f"Number of essays with more than 1,000 words: {essays_over_1k}")
    print(f"Long essays copied to: {long_essays_dir}")
    
    # print count of essays processed
    print(f"Total essays processed: {len(results)}")

if __name__ == "__main__":
    process_essays()
