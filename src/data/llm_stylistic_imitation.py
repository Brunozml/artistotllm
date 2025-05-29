# %%
# -*- coding: utf-8 -*-

# %% [markdown]
"""
# Text Similarity Metrics

This script generates plausible continuations to essays and texts from different sources.
It then calculates various text similarity metrics for comparing
original texts, predictions, and test samples.

## Setup and Data Loading
"""

# %%
import os
import sys
import re
import string
import numpy as np
import pandas as pd
import nltk
from collections import Counter
from typing import Dict, Tuple, Iterable, List, Any, Union
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# %%  Set up path
current_dir = os.getcwd()
print("Current working directory:", current_dir)


# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# %% [markdown] 

# Comment out this code to skip the LLM generation step. - Needs GPUs!

"""
# Generate LLM Text

"""
model_id = "Qwen/Qwen3-4B"          # or any 8-B variant, e.g. -AWQ, -FP8, GGUF etc.
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(
               model_id,
               device_map="auto",           # GPU if you have one
               torch_dtype="auto"           # fp16/bf16 automatically
           )
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

nlp_results = pd.DataFrame(columns=['Author', 'Title', 'y_train', 'y_pred', 'y_test'])

def remove_prefix(original, full):
    if full.startswith(original):
        return full[len(original):].lstrip()  # Remove leading spaces/newlines
    else:
        raise ValueError("The first text is not a prefix of the second.")

def split_text_by_sentences(text, word_limit):
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    train_text = ''
    word_count = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        if word_count + len(sentence_words) > word_limit:
            break
        train_text += sentence + ' '
        word_count += len(sentence_words)

    # Now get the next 500 words after the train_text
    remaining_text = text[len(train_text):].strip()
    next_words = ' '.join(remaining_text.split()[:500])

    return train_text.strip(), next_words.strip()
base_path = "data/raw/"
authors = ["Twain", "Graham", "Krugman", "Hansen"]

for author in authors:
    folder_path = os.path.join(base_path, author)
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            # Try different encodings with error handling
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'ascii', 'utf-16']
            text = ""
            for encoding in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                        print(f"Successfully decoded {filename} using {encoding}")
                        break  # If successful, break out of the encoding loop
                except UnicodeDecodeError as e:
                    print(f"Failed to decode {filename} with {encoding}: {str(e)}")
                    if encoding == encodings_to_try[-1]:  # If this was the last encoding to try
                        print(f"Failed to decode {filename} with all attempted encodings")
                        text = ""  # Set empty text or handle the error as needed
                    continue  # Try the next encoding
            
            train_text, test_text = split_text_by_sentences(text, 500)
            # Do whatever you want with train_text and test_text
            print(f'Processed {filename} from {author} using {encoding} encoding')

                pred = generator(
                    train_text,
                    # generation control
                    max_new_tokens=500,      # ← exactly the number you asked for
                    temperature=0.8,         # creativity (0–2); lower is safer
                    top_p=0.9,               # nucleus sampling
                    do_sample=True,          # sampling instead of greedy
                    eos_token_id=tokenizer.eos_token_id,   # stop at end-of-text if it appears
                )
                y_pred = pred[0]["generated_text"]
                result = remove_prefix(train_text, y_pred)
                new_row = pd.DataFrame([{'Author': author, 'Title': filename, 'y_pred': result, 'y_train': train_text, 'y_test': test_text}])
                nlp_results = pd.concat([nlp_results, new_row], ignore_index=True)

                print(nlp_results.shape)
                print(nlp_results)
                
# %%
df = result.copy()
def word_count(text):
    return len(str(text).split())

# Calculate word count for each column
df['y_train_word_count'] = df['y_train'].apply(word_count)
df['y_test_word_count'] = df['y_test'].apply(word_count)
df['y_pred_word_count'] = df['y_pred'].apply(word_count)

# Sum the word counts
df['total_word_count'] = df['y_train_word_count'] + df['y_test_word_count']

# Filter the DataFrame
filtered_df = df[df['total_word_count'] >= 900]
filtered_df

# %%
DATA_PATH = os.path.join(current_dir, "final_dataset.csv")
filtered_df.to_csv("final_dataset.csv", index=None)

# %% [markdown]
"""
# Metrics Calculation
This section contains functions to calculate various text similarity metrics.

## Metrics Overview
This notebook calculates the following text similarity metrics:
- **Stopwords Analysis**: Distribution and similarity of stopwords.
- **Punctuation Analysis**: Distribution and similarity of punctuation marks.
- **Type-Token Ratio (TTR)**: Lexical diversity metrics including TTR, MATTR, and MTLD.
- **Sentence Transformers**: Semantic similarity using pre-trained sentence embeddings.
- **Part-of-Speech (POS)**: Distribution and similarity of POS tags.
- **Sentence Length**: Similarity based on sentence length distributions.



"""
# You can safely run this code to generate metrics 

DATA_PATH = os.path.join(current_dir, "final_dataset.csv")

# Download necessary NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# %% [markdown]
"""
## Loading the Dataset

We load the dataset containing text samples for comparison.
"""

# %%
df = pd.read_csv(DATA_PATH)

# %%
"""
## 1. Stopwords Analysis

Analyzes the distribution of stopwords in texts and compares them.
"""

# Shared resources
ENGLISH_STOPWORDS: List[str] = stopwords.words("english")
EN_STOP = set(ENGLISH_STOPWORDS)          # fast membership test
TOKEN_PATTERN = re.compile(r"\b\w+\b").findall

def extract_stopwords(text: str | None) -> List[str]:
    """Return the stop-words that actually appear in *text* (lower-cased)."""
    if not text:
        return []
    tokens = TOKEN_PATTERN(str(text).lower())
    return [tok for tok in tokens if tok in EN_STOP]


def get_stopword_distribution(text: str | None) -> Dict[str, float]:
    """Normalized frequency of each stop-word + a 'density' feature."""
    stopword_tokens = extract_stopwords(text)
    counts = Counter(stopword_tokens)
    total = sum(counts.values()) or 1                         # avoid /0
    # vector with *all* standard stop-words (missing ones → 0)
    dist = {w: counts.get(w, 0) / total for w in ENGLISH_STOPWORDS}

    # density = stop-words per word
    word_cnt = len(word_tokenize(str(text or "").lower()))
    dist["density"] = len(stopword_tokens) / word_cnt if word_cnt else 0
    return dist


def stopwords_similarity(text1: str | None,
                         text2: str | None) -> Tuple[float, Dict]:
    """Cosine similarity between the two distributions + rich diagnostics."""
    d1, d2 = get_stopword_distribution(text1), get_stopword_distribution(text2)

    features = ENGLISH_STOPWORDS + ["density"]
    v1 = np.array([d1[f] for f in features])
    v2 = np.array([d2[f] for f in features])

    sim = (np.dot(v1, v2) /
           (np.linalg.norm(v1) * np.linalg.norm(v2))) if v1.any() and v2.any() else 0.0

    # optional detail payload (trim to first 20 stop-words for readability)
    most1, most2 = Counter(extract_stopwords(text1)).most_common(10), \
                   Counter(extract_stopwords(text2)).most_common(10)
    top_overlap = len({w for w, _ in most1}.intersection({w for w, _ in most2}))

    details = {
        "text1_stats": {"most_common": most1, "density": d1["density"]},
        "text2_stats": {"most_common": most2, "density": d2["density"]},
        "top10_overlap": top_overlap,
        "similarity": sim
    }
    return sim, details


def add_stopword_columns(df: pd.DataFrame,
                         text_cols: Iterable[str] = ("y_train", "y_pred", "y_test")
                         ) -> pd.DataFrame:
    """Add a <col>_stopwords list column for every *text_cols* entry."""
    for col in text_cols:
        df[f"{col}_stopwords"] = df[col].apply(extract_stopwords)
    return df


def add_stopword_similarity(df: pd.DataFrame,
                            pairs: Iterable[Tuple[str, str]] = (
                                ("y_train", "y_pred"),
                                ("y_pred", "y_test"),
                                ("y_train", "y_test")
                            ),
                            keep_details: bool = False
                            ) -> pd.DataFrame:
    """
    For each (colA, colB) pair, append a cosine-similarity score column.

    • Column is named  '<colA>_vs_<colB>_stop_sim'.  
    • If *keep_details* is True, a second column with the same stem
      plus '_details' is added containing the verbose diagnostics dict.
    """
    for a, b in pairs:
        sim_col = f"{a}_vs_{b}_stop_sim"
        if keep_details:
            det_col = f"{a}_vs_{b}_stop_sim_details"

            def _pair(row):
                sim, det = stopwords_similarity(row[a], row[b])
                return pd.Series({sim_col: sim, det_col: det})

            df[[sim_col, det_col]] = df.apply(_pair, axis=1)
        else:
            df[sim_col] = df.apply(lambda r: stopwords_similarity(r[a], r[b])[0],
                                   axis=1)
    return df

# %%
# Apply stopwords analysis
df = add_stopword_columns(df)
df = add_stopword_similarity(df, keep_details=False)

# %%
"""
## 2. Punctuation Analysis

Analyzes the distribution of punctuation marks in texts and compares them.
"""

# Shared resources
PUNCTUATION_CHARS: List[str] = list(string.punctuation)        # 32 ASCII marks
PUNCT_SET = set(PUNCTUATION_CHARS)
PUNCT_PATTERN = re.compile(f"[{re.escape(string.punctuation)}]")  # matches ONE mark


def extract_punctuation(text: str | None) -> List[str]:
    """Return *every* punctuation mark that appears in *text* (one per hit)."""
    if not text:
        return []
    return PUNCT_PATTERN.findall(str(text))


def get_punctuation_distribution(text: str | None) -> Dict[str, float]:
    """
    Normalised frequency of each ASCII punctuation mark (+ 'density').

    • Density = punctuation marks per *character* (not word) so it stays
      meaningful even for very short snippets such as "Hi!".
    """
    tokens = extract_punctuation(text)
    counts = Counter(tokens)
    total = sum(counts.values()) or 1

    dist = {ch: counts.get(ch, 0) / total for ch in PUNCTUATION_CHARS}

    char_count = len(str(text or ""))
    dist["density"] = len(tokens) / char_count if char_count else 0
    return dist


def punctuation_similarity(text1: str | None,
                           text2: str | None) -> Tuple[float, Dict]:
    """Cosine-similarity of punctuation distributions + handy diagnostics."""
    d1, d2 = (get_punctuation_distribution(text1),
              get_punctuation_distribution(text2))

    feats = PUNCTUATION_CHARS + ["density"]
    v1 = np.array([d1[f] for f in feats])
    v2 = np.array([d2[f] for f in feats])

    sim = (np.dot(v1, v2) /
           (np.linalg.norm(v1) * np.linalg.norm(v2))) if v1.any() and v2.any() else 0.0

    most1, most2 = Counter(extract_punctuation(text1)).most_common(5), \
                   Counter(extract_punctuation(text2)).most_common(5)
    overlap = len({ch for ch, _ in most1}.intersection({ch for ch, _ in most2}))

    details = {
        "text1_stats": {"most_common": most1, "density": d1["density"]},
        "text2_stats": {"most_common": most2, "density": d2["density"]},
        "top5_overlap": overlap,
        "similarity": sim
    }
    return sim, details


def add_punctuation_columns(df: pd.DataFrame,
                            text_cols: Iterable[str] = ("y_train",
                                                        "y_pred",
                                                        "y_test")
                            ) -> pd.DataFrame:
    """Add a <col>_punct list column for every *text_cols* entry."""
    for col in text_cols:
        df[f"{col}_punct"] = df[col].apply(extract_punctuation)
    return df


def add_punctuation_similarity(df: pd.DataFrame,
                               pairs: Iterable[Tuple[str, str]] = (
                                   ("y_train", "y_pred"),
                                   ("y_pred", "y_test"),
                                   ("y_train", "y_test")
                               ),
                               keep_details: bool = False
                               ) -> pd.DataFrame:
    """
    Append cosine-similarity columns for each (colA, colB) pair based on punctuation.

    • Column:  '<colA>_vs_<colB>_punct_sim'  
    • If *keep_details* is True, also add '<pair>_punct_sim_details' with diagnostics.
    """
    for a, b in pairs:
        sim_col = f"{a}_vs_{b}_punct_sim"

        if keep_details:
            det_col = f"{a}_vs_{b}_punct_sim_details"

            def _pair(row):
                sim, det = punctuation_similarity(row[a], row[b])
                return pd.Series({sim_col: sim, det_col: det})

            df[[sim_col, det_col]] = df.apply(_pair, axis=1)
        else:
            df[sim_col] = df.apply(lambda r: punctuation_similarity(r[a], r[b])[0],
                                   axis=1)
    return df

# %%
# Apply punctuation analysis
df = add_punctuation_columns(df)
df = add_punctuation_similarity(df, keep_details=False)

# %%
"""
## 3. Type-Token Ratio (TTR) Similarity

Measures lexical diversity using type-token ratio metrics and compares texts.
"""

def preprocess_text(text: str, remove_stopwords: bool = False) -> List[str]:
    """
    Preprocess text by tokenizing, lowercasing, and optionally removing stopwords.
    
    Args:
        text (str): The input text to preprocess
        remove_stopwords (bool): Whether to remove stopwords
        
    Returns:
        List[str]: Preprocessed tokens
    """
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove non-alphabetic tokens
    tokens = [token for token in tokens if token.isalpha()]
    
    # Optionally remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    return tokens

def calculate_ttr(tokens: List[str]) -> float:
    """
    Calculate the Type-Token Ratio (TTR).
    
    Args:
        tokens (List[str]): List of tokens from the text
        
    Returns:
        float: The Type-Token Ratio value
    """
    if not tokens:
        return 0
    
    n_types = len(set(tokens))  # Number of unique words
    n_tokens = len(tokens)      # Total number of words
    
    return n_types / n_tokens

def moving_average_ttr(tokens: List[str], window_size: int = 100) -> float:
    """
    Calculate Moving-Average Type-Token Ratio (MATTR).
    
    Args:
        tokens (List[str]): List of tokens from the text
        window_size (int): Size of the sliding window
        
    Returns:
        float: The MATTR value
    """
    if len(tokens) < window_size:
        return calculate_ttr(tokens)
    
    # Calculate TTR for each window and take the average
    ttrs = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i+window_size]
        ttrs.append(calculate_ttr(window))
    
    return sum(ttrs) / len(ttrs)

def mtld(tokens: List[str], threshold: float = 0.72) -> float:
    """
    Calculate Measure of Textual Lexical Diversity (MTLD).
    
    Args:
        tokens (List[str]): List of tokens from the text
        threshold (float): The TTR threshold for factor count
        
    Returns:
        float: The MTLD value
    """
    if len(tokens) < 50:  # Too short for reliable MTLD
        return 0
    
    def mtld_pass(tokens, threshold):
        # Forward pass
        factors = 0
        types_so_far = set()
        token_count = 0
        
        for token in tokens:
            token_count += 1
            types_so_far.add(token)
            ttr = len(types_so_far) / token_count
            
            if ttr <= threshold:
                factors += 1
                types_so_far = set()
                token_count = 0
        
        if token_count > 0:
            ttr = len(types_so_far) / token_count
            partial_factor = (1 - ttr) / (1 - threshold)
            factors += partial_factor
        
        return len(tokens) / factors if factors > 0 else 0
    
    # Calculate MTLD as the average of forward and backward passes
    forward = mtld_pass(tokens, threshold)
    backward = mtld_pass(tokens[::-1], threshold)
    
    return (forward + backward) / 2

def ttr_similarity(text1: str, text2: str, include_stopwords: bool = True) -> Tuple[float, Dict]:
    """
    Compare two texts based on their lexical diversity (TTR) metrics.
    
    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare
        include_stopwords (bool): Whether to include stopwords in the analysis
        
    Returns:
        Tuple[float, Dict]: A tuple containing:
            - A similarity score between 0-1
            - A dictionary with detailed metrics
    """
    # Preprocess texts
    tokens1 = preprocess_text(text1, remove_stopwords=not include_stopwords)
    tokens2 = preprocess_text(text2, remove_stopwords=not include_stopwords)
    
    # Calculate basic TTR for both texts
    ttr1 = calculate_ttr(tokens1)
    ttr2 = calculate_ttr(tokens2)
    
    # Calculate MATTR for both texts
    mattr1 = moving_average_ttr(tokens1)
    mattr2 = moving_average_ttr(tokens2)
    
    # Calculate MTLD for both texts
    mtld1 = mtld(tokens1)
    mtld2 = mtld(tokens2)
    
    # Calculate similarity scores (1 - normalized absolute difference)
    ttr_sim = 1 - abs(ttr1 - ttr2) / max(ttr1, ttr2) if max(ttr1, ttr2) > 0 else 1
    mattr_sim = 1 - abs(mattr1 - mattr2) / max(mattr1, mattr2) if max(mattr1, mattr2) > 0 else 1
    mtld_sim = 1 - abs(mtld1 - mtld2) / max(mtld1, mtld2) if max(mtld1, mtld2) > 0 else 1
    
    # Calculate overall similarity (weighted average)
    overall_sim = (ttr_sim * 0.3) + (mattr_sim * 0.4) + (mtld_sim * 0.3)
    
    # Prepare detailed output
    details = {
        "ttr": {
            "text1": ttr1, 
            "text2": ttr2, 
            "similarity": ttr_sim
        },
        "mattr": {
            "text1": mattr1, 
            "text2": mattr2, 
            "similarity": mattr_sim
        },
        "mtld": {
            "text1": mtld1, 
            "text2": mtld2, 
            "similarity": mtld_sim
        },
        "text1_stats": {
            "tokens": len(tokens1), 
            "unique_tokens": len(set(tokens1)),
            "lexical_density": len(set(tokens1)) / len(tokens1) if tokens1 else 0
        },
        "text2_stats": {
            "tokens": len(tokens2), 
            "unique_tokens": len(set(tokens2)),
            "lexical_density": len(set(tokens2)) / len(tokens2) if tokens2 else 0
        }
    }
    
    return overall_sim, details

def add_ttr_similarity(df: pd.DataFrame,
                       pairs: Iterable[Tuple[str, str]] = (
                           ("y_train", "y_pred"),
                           ("y_pred",  "y_test"),
                           ("y_train", "y_test")
                       ),
                       include_stopwords: bool = True,
                       keep_details: bool = False
                       ) -> pd.DataFrame:
    """
    Append a lexical-diversity similarity column for each (colA, colB) pair.

    Parameters
    ----------
    df : pandas.DataFrame
        Your frame containing the text columns.
    pairs : iterable[tuple[str, str]]
        Column name pairs to compare (default = the three pair-wise combos
        of 'y_train', 'y_pred', 'y_test').
    include_stopwords : bool
        Passed straight through to `ttr_similarity()`.
    keep_details : bool
        • False  → add only '<colA>_vs_<colB>_ttr_sim' (float 0–1).  
        • True   → also add '<pair>_ttr_sim_details' with the full metrics dict.

    Returns
    -------
    pandas.DataFrame
        The same frame, with the new similarity (and optional details) columns.
    """
    for a, b in pairs:
        sim_col = f"{a}_vs_{b}_ttr_sim"

        if keep_details:
            det_col = f"{a}_vs_{b}_ttr_sim_details"

            def _compare(row):
                sim, det = ttr_similarity(row[a],
                                          row[b],
                                          include_stopwords=include_stopwords)
                return pd.Series({sim_col: sim, det_col: det})

            df[[sim_col, det_col]] = df.apply(_compare, axis=1)

        else:
            df[sim_col] = df.apply(
                lambda r: ttr_similarity(r[a],
                                         r[b],
                                         include_stopwords=include_stopwords)[0],
                axis=1
            )
    return df

# %%
# Apply TTR similarity analysis
df = add_ttr_similarity(df, keep_details=False)

# %%
"""
## 4. Sentence Transformers Similarity

Uses pre-trained sentence embeddings to calculate semantic similarity between texts.
"""

# Initialize the model once for reuse
_SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")   # ~80 MB, tiny + fast

def transformer_similarity(text1: str | None,
                          text2: str | None,
                          model: SentenceTransformer = _SBERT_MODEL
                          ) -> Tuple[float, dict]:
    """
    Cosine similarity between the two sentence-level embeddings.
    Returns (similarity, small-diagnostics-dict).
    """
    text1 = text1 or ""
    text2 = text2 or ""
    emb1, emb2 = model.encode([text1, text2], normalize_embeddings=True)

    # cosine on **already L2-normalised** vectors = simple dot product
    similarity = float(np.dot(emb1, emb2))

    details = {
        "embedding1_sample": emb1[:5].tolist(),   # first 5 dims for sanity-check
        "embedding2_sample": emb2[:5].tolist()
    }
    return similarity, details

def add_transformer_similarity(df: pd.DataFrame,
                               pairs: Iterable[Tuple[str, str]] = (
                                   ("y_train", "y_pred"),
                                   ("y_pred",  "y_test"),
                                   ("y_train", "y_test")
                               ),
                               keep_details: bool = False,
                               model: SentenceTransformer = _SBERT_MODEL
                               ) -> pd.DataFrame:
    """
    Append semantic similarity columns computed with a Sentence-BERT model.

    • For each (colA, colB) you get '<colA>_vs_<colB>_sbert_sim'  (float, –1‥1).  
      (Because embeddings are normalised, scores are usually 0‥1 for
       "reasonably related" English texts; negatives mean strong dissimilarity.)

    • With *keep_details=True* a companion
      '<pair>_sbert_sim_details' column is added containing the two
      1 024-D vectors' first five dimensions.

    • Pass your own *model* if you want a different checkpoint
      (e.g. multilingual or domain-specific).
    """
    for a, b in pairs:
        sim_col = f"{a}_vs_{b}_sbert_sim"

        if keep_details:
            det_col = f"{a}_vs_{b}_sbert_sim_details"

            def _compare(row):
                sim, det = transformer_similarity(row[a], row[b], model)
                return pd.Series({sim_col: sim, det_col: det})

            df[[sim_col, det_col]] = df.apply(_compare, axis=1)

        else:
            df[sim_col] = df.apply(
                lambda r: transformer_similarity(r[a], r[b], model)[0],
                axis=1
            )
    return df

# %%
# Apply transformer similarity analysis
df = add_transformer_similarity(df, keep_details=False)

# %%
"""
## 5. Part-of-Speech (POS) Similarity

Analyzes the distribution of POS tags in texts and compares them.
"""

def get_pos_distribution(text: str | None) -> Dict[str, float]:
    """
    Return a normalised frequency table of Penn-Treebank POS tags.
    """
    tokens = word_tokenize(str(text or "").lower())
    tags = pos_tag(tokens)

    counts = {}
    for _, tag in tags:
        counts[tag] = counts.get(tag, 0) + 1

    total = sum(counts.values()) or 1
    return {tag: c / total for tag, c in counts.items()}

def pos_similarity(text1: str | None,
                  text2: str | None
                  ) -> Tuple[float, Dict]:
    """
    Cosine similarity between the two POS-tag distributions.
    """
    dist1, dist2 = get_pos_distribution(text1), get_pos_distribution(text2)
    all_tags = set(dist1) | set(dist2)

    v1 = np.array([dist1.get(tag, 0.0) for tag in all_tags])
    v2 = np.array([dist2.get(tag, 0.0) for tag in all_tags])

    sim = (np.dot(v1, v2) /
           (np.linalg.norm(v1) * np.linalg.norm(v2))) if v1.any() and v2.any() else 0.0

    return sim, {"text1_pos": dist1, "text2_pos": dist2}

def add_pos_similarity(df: pd.DataFrame,
                       pairs: Iterable[Tuple[str, str]] = (
                           ("y_train", "y_pred"),
                           ("y_pred",  "y_test"),
                           ("y_train", "y_test")
                       ),
                       keep_details: bool = False
                       ) -> pd.DataFrame:
    """
    Append POS-distribution similarity columns.

    • For each (colA, colB) → `<colA>_vs_<colB>_pos_sim` (float 0–1).  
    • If *keep_details* is True, also add
      `<pair>_pos_sim_details` with both normalised distributions.
    """
    for a, b in pairs:
        sim_col = f"{a}_vs_{b}_pos_sim"

        if keep_details:
            det_col = f"{a}_vs_{b}_pos_sim_details"

            def _row(row):
                sim, det = pos_similarity(row[a], row[b])
                return pd.Series({sim_col: sim, det_col: det})

            df[[sim_col, det_col]] = df.apply(_row, axis=1)
        else:
            df[sim_col] = df.apply(
                lambda r: pos_similarity(r[a], r[b])[0],
                axis=1
            )
    return df

# %%
# Apply POS similarity analysis
df = add_pos_similarity(df, keep_details=False)

# %%
"""
## 6. Sentence Length Similarity

Compares texts based on their sentence length distributions.
"""

def calculate_sentence_stats(text: str) -> Dict[str, float]:
    """
    Return average-, median-, std-sentence length plus raw lengths list.
    """
    text = str(text or "").strip()

    # crude sentence split (period / exclam / question)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences]

    avg = sum(lengths) / len(sentences) if sentences else 0.0

    return {
        "avg_sentence_length": avg,
        "median_sentence_length": float(np.median(lengths)) if lengths else 0.0,
        "std_deviation": float(np.std(lengths)) if lengths else 0.0,
        "total_sentences": len(sentences),
        "sentence_lengths": lengths,
    }

def sentence_length_similarity(text1: str,
                              text2: str
                              ) -> Tuple[float, Dict]:
    """Similarity = min(avg1, avg2) / max(avg1, avg2) (range 0–1)."""
    stats1, stats2 = (calculate_sentence_stats(text1),
                      calculate_sentence_stats(text2))

    avg1, avg2 = stats1["avg_sentence_length"], stats2["avg_sentence_length"]

    if avg1 == avg2 == 0:
        sim = 1.0
    elif avg1 == 0 or avg2 == 0:
        sim = 0.0
    else:
        sim = min(avg1, avg2) / max(avg1, avg2)

    details = {
        "text1_stats": stats1,
        "text2_stats": stats2,
        "difference": avg2 - avg1,
    }
    return sim, details

def add_sentence_length_similarity(df: pd.DataFrame,
                                  pairs: Iterable[Tuple[str, str]] = (
                                      ("y_train", "y_pred"),
                                      ("y_pred",  "y_test"),
                                      ("y_train", "y_test"),
                                  ),
                                  keep_details: bool = False
                                  ) -> pd.DataFrame:
    """
    Append per-row sentence-length similarity columns.

    • Each (colA, colB) pair adds
        '<colA>_vs_<colB>_sentlen_sim'            (float 0–1)

    • If *keep_details* is True a companion
        '<pair>_sentlen_sim_details'
      column contains the full stats dict.
    """
    for a, b in pairs:
        sim_col = f"{a}_vs_{b}_sentlen_sim"

        if keep_details:
            det_col = f"{a}_vs_{b}_sentlen_sim_details"

            def _row(row):
                sim, det = sentence_length_similarity(row[a], row[b])
                return pd.Series({sim_col: sim, det_col: det})

            df[[sim_col, det_col]] = df.apply(_row, axis=1)

        else:
            df[sim_col] = df.apply(
                lambda r: sentence_length_similarity(r[a], r[b])[0],
                axis=1
            )
    return df

# %%
# Apply sentence length similarity analysis
df = add_sentence_length_similarity(df, keep_details=False)

# %%
"""
## Results Summary

Overview of the calculated metrics and their significance.
"""
# Show a summary of all calculated metrics
print("Calculated metrics for each text pair:")
metric_cols = [col for col in df.columns if "_sim" in col]
print(df[metric_cols].describe())

# Sample display of a few rows with all metrics
print("\nSample rows with calculated metrics:")
print(df[['y_train', 'y_pred', 'y_test'] + metric_cols].head())

# %%
"""
## Save Results

Save the DataFrame with all calculated metrics to a CSV file.
"""
# Save the enriched dataset if needed
OUTPUT_PATH = os.path.join(current_dir, "text_metrics_results.csv")
df.to_csv(OUTPUT_PATH, index=False)
print(f"Results saved to {OUTPUT_PATH}")

# %%
"""
## Visualization (Optional)

Add visualization code here to plot the distributions of the metrics.
"""
# Example visualization code (uncomment to use)
'''
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.boxplot(data=df[metric_cols])
plt.title('Distribution of Text Similarity Metrics')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
'''
