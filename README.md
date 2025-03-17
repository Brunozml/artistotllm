# ArtistotLLM

A framework for fine-tuning language models to capture the writing style and reasoning patterns of historical philosophers, starting with Aristotle.

## Project Structure

- `data/`: Contains raw and processed text data
- `src/`: Source code organized by functionality
- `configs/`: Configuration files for models and training
- `notebooks/`: Jupyter notebooks for exploration
- `tests/`: Unit tests

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

1. Place your raw text data in `data/raw/`
2. Configure model and training parameters in `configs/`
3. Run the training pipeline:

```bash
python src/models/training.py
```

## Development

This project follows a modular architecture where each component is responsible for a specific task:

- Data processing: Text cleaning, tokenization
- Model management: Model setup, training loops
- Evaluation: Metrics and analysis tools

## License

MIT License - see LICENSE file for details
