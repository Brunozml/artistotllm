# ArtistotLLM

A framework for fine-tuning language models to capture the writing style and reasoning patterns of historical philosophers, starting with [Aristotle](https://www.gutenberg.org/ebooks/author/2747) and [Sir Arthur Conan Doyle](https://www.gutenberg.org/ebooks/author/69).

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

## Sir Arthur Conan Doyle 

Sir Arthur Conan Doyle (1859 - 1930) wrote a significant number of books across various genres. His bibliography includes:

- **Novels:** 22 novels, including those featuring Sherlock Holmes and other standalone works like "The Lost World" and "The White Company"[^5][^7].
- **Short Story Collections:** 16 collections, many of which are part of the Sherlock Holmes series[^5].
- **Plays:** 14 plays[^5].
- **Spiritualist and Paranormal Books:** 13 books, reflecting his interest in spiritualism[^5].
- **Poems:** 4 collections of poetry[^5].

In total, Doyle published over 200 stories and articles throughout his career[^5]. His most famous works remain the Sherlock Holmes stories, which include four novels and 58 short stories[^1][^8].

<div style="text-align: center">⁂</div>

[^1]: https://en.wikipedia.org/wiki/Arthur_Conan_Doyle

[^2]: https://www.arthur-conan-doyle.com/index.php/The_62_Sherlock_Holmes_stories_written_by_Arthur_Conan_Doyle

[^3]: https://www.arthur-conan-doyle.com/index.php/Sir_Arthur_Conan_Doyle:Complete_Works

[^4]: https://www.fantasticfiction.com/d/sir-arthur-conan-doyle/

[^5]: https://www.sherlockian.net/investigating/acdbibliography/

[^6]: https://en.wikipedia.org/wiki/Arthur_Conan_Doyle_bibliography

[^7]: https://arthurconandoyle.co.uk/author

[^8]: https://www.britannica.com/biography/Arthur-Conan-Doyle

[^9]: https://en.wikipedia.org/wiki/Arthur_Conan_Doyle


## License

MIT License - see LICENSE file for details
