# TinyDetector: Lightweight Text Safety Classifier

TinyDetector is a small PyTorch-based content safety classifier that detects
harmful text (e.g., toxicity, hate speech) in real time.

It is designed as a personal project

## Planned Task

Classify short text into three classes:

- `safe`
- `toxic`
- `hate`

using a fine-tuned transformer model (e.g., DistilBERT).

## Dataset (planned)

I plan to use a subset of a public toxic comment dataset (such as the
Jigsaw Toxic Comment Classification dataset) and map the original labels
to three coarse classes:

- `safe`: comments without toxicity labels
- `toxic`: general toxicity / insult / obscene
- `hate`: identity-based hate and related labels

Details and preprocessing steps will be documented as the project evolves.

## Repo Structure

- `src/`: model, datasets, training, evaluation, inference
- `data/`: raw and processed data (not tracked in git)
- `notebooks/`: EDA and failure analysis
- `tests/`: basic tests for datasets and inference
- `app/`: (to be added) demo UI, likely using Gradio
