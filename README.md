# ArtistotLLM

A research framework for analyzing how well Large Language Models (LLMs) can imitate the writing style of specific authors. This project focuses on measuring the similarity between LLM-generated continuations of texts and the actual continuations by the original authors.

## Research Question

**To what extent can LLMs imitate the stylistic writing of an author?**

## Project Overview

As LLMs advance, they become increasingly capable of producing text that resembles human writing. This project investigates how accurately these models can replicate the unique stylistic features of individual authors, including:

- Vocabulary choices
- Sentence structure
- Use of punctuation
- Tone and rhetorical strategies

We analyze texts from four selected authors (Twain, Graham, Krugman, and Hansen) and design comprehensive metrics to measure the similarity between LLM-generated continuations and the authors' actual writing styles.

## Project Structure

- `data/`: Contains raw author texts and processed datasets
  - `raw/`: Original texts from different authors
  - `processed/`: Prepared datasets for analysis
- `src/`: Source code organized by functionality
  - `data/`: Text processing and analysis scripts
  - `evaluation/`: Metrics for measuring text similarity
  - `utils/`: Helper functions
- `configs/`: Configuration files for models and experiments
- `notebooks/`: Jupyter notebooks for data exploration and visualization
- `results/`: Analysis outputs and visualizations
  - `essay_analysis/`: Comparisons of essays
  - `metrics_analysis/`: Detailed metrics results

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Place author text data in `data/raw/{Author}/`
2. Run the LLM text generation script:

```bash
python src/data/llm_stylistic_imitation.py
```

3. Analyze the results with various metrics:

```bash
python src/data/analyze_all_metrics.py
```

## Text Similarity Metrics

This project implements multiple metrics to comprehensively evaluate how well LLM-generated text imitates an author's style:

1. **Stopwords Analysis**: Distribution and similarity of stopwords
2. **Punctuation Analysis**: Patterns of punctuation usage
3. **Type-Token Ratio (TTR)**: Lexical diversity metrics (TTR, MATTR, MTLD)
4. **Sentence Transformers**: Semantic similarity using embeddings
5. **Part-of-Speech (POS)**: Distribution of POS tags
6. **Sentence Length**: Analysis of sentence structure

## Development

This project follows a modular architecture where each component is responsible for a specific task:

- **Data processing**: Text cleaning, tokenization, and prompt generation
- **Metric implementation**: Various text similarity measurements
- **Evaluation**: Comparative analysis across different authors and texts

## Authors Analyzed

This project analyzes texts from four distinct authors with recognizable writing styles:

1. **Mark Twain**: American writer known for his witty, satirical style and use of regional dialects
2. **Paul Graham**: Programmer and essayist with a clear, direct style focused on technology and startups
3. **Paul Krugman**: Economist and columnist with a analytical approach to economic topics
4. **Robin Hansen**: Academic with a unique writing style addressing future technology and society

## Research Methodology

Our approach involves:

1. **Corpus Collection**: Gathering text samples from each author
2. **Text Segmentation**: Splitting texts into training portions (~500 words) and test portions
3. **LLM Generation**: Prompting an LLM (Qwen3-4B) to continue the training portion
4. **Metric Design**: Creating comprehensive metrics for stylistic comparison
5. **Comparative Analysis**: Measuring how well the LLM-generated text matches the author's actual continuation

## Results

The results directory contains CSV and Excel files with detailed metrics comparing original and LLM-generated texts. Visualizations and analyses can be found in the notebooks directory.

## Motivation

This research helps us understand:

- Which aspects of writing style LLMs can effectively imitate
- Where they still struggle to capture an author's unique voice
- How to better detect AI-generated content attempting to mimic human authors
- The evolution of stylistic imitation capabilities in language models

## License

MIT License - see LICENSE file for details
