# NLP N-gram Analysis

## Overview

Tokenization analysis and n-gram language modeling on Wall Street Journal text data.

- **Part A** — Compares three tokenization methods (NLTK, spaCy, BERT), computes corpus statistics (tokens, types, TTR, hapax legomena), and validates Zipf's law with log-log frequency plots.
- **Part B** — Builds bigram and trigram language models with add-k smoothing on the Penn Treebank corpus. Evaluates perplexity across preprocessing modes (original, lowercase, abstract digits) and generates new sentences via weighted sampling.

## Project Structure

```
.
├── assignment_1.ipynb       # Unified Jupyter notebook (main deliverable)
├── scripts/
│   ├── part_a.py            # Part A: tokenization, statistics, Zipf's law
│   └── part_b.py            # Part B: n-gram models, perplexity, generation
├── src/nlp_ngram/           # Core reusable library (data loading, base model)
│   ├── __init__.py
│   ├── data.py
│   ├── ngram.py
│   └── cli.py
├── tests/                   # Unit tests
├── wsj_untokenized.txt      # WSJ corpus for Part A
├── requirements.txt         # Python dependencies
└── pyproject.toml           # Package metadata
```

## Setup

```bash
conda create -n nlp-ngram python=3.11 -y
conda activate nlp-ngram
pip install -r requirements.txt
```

## Running

### Jupyter Notebook (recommended)

The notebook `assignment_1.ipynb` runs the full pipeline for both parts with tables, plots, and commentary sections.

```bash
jupyter notebook assignment_1.ipynb
```

### Standalone Scripts

Each script can also be run independently from the terminal:

**Part A** — Tokenization & Zipf's Law:
```bash
python scripts/part_a.py                          # uses default wsj_untokenized.txt
python scripts/part_a.py --corpus other_file.txt  # custom corpus path
```

Outputs: statistics table, sentence comparison, top types, Zipf's law plot (`zipf_plot.png`).

**Part B** — N-gram Language Models:
```bash
python scripts/part_b.py
```

Outputs: perplexity table (4 models x 3 preprocessing modes), generated sentences.

### Tests

```bash
pytest
```

## Dependencies

- `nltk` — tokenization (Part A) and treebank corpus (Part B)
- `spacy` + `en_core_web_sm` — tokenization (Part A)
- `transformers` + `torch` — BERT tokenizer (Part A)
- `matplotlib`, `pandas`, `numpy` — plotting and tables
- `jupyter` — notebook execution
