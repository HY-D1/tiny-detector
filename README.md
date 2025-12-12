# TinyDetector: Lightweight Text Safety Classifier

TinyDetector is a small PyTorch-based content safety classifier that flags harmful text (e.g., toxicity and identity-based hate) using a fine-tuned transformer model.

The project is intentionally lightweight and modular: it includes data preparation, training, evaluation, and inference utilities, plus notebooks for exploration and failure analysis.

## What it does

TinyDetector maps input text into one of three labels:

- `safe`
- `toxic`
- `hate`

It provides:

- a training pipeline to fine-tune a transformer model (DistilBERT by default),
- an evaluation script that generates metrics and a confusion matrix,
- an inference API for running predictions on custom text,
- notebooks for exploratory analysis and error inspection.

## Dataset and label mapping

This project uses the **Jigsaw Toxic Comment Classification** dataset as the source of labeled comments. The original dataset includes multiple binary labels such as:

- `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

These are mapped into three coarse classes:

- `safe` (0): no toxic labels
- `hate` (2): `identity_hate == 1`
- `toxic` (1): any other toxic label (e.g., `toxic`, `insult`, `obscene`, `threat`, `severe_toxic`) without identity hate

A stratified train/validation split is created and saved as:

- `data/processed/train.csv`
- `data/processed/val.csv`

## Installation and environment

### Docker (recommended)

Build the dev image:

```bash
docker build -t tiny-detector:dev .
```

Start a container with the repo mounted:

```base
docker run --rm -it \
  -v "$(pwd)":/app \
  -p 8888:8888 \
  tiny-detector:dev \
  bash
```

### Training

From inside the container (or your local environment):
```bash
python -m src.train
```

This:

- loads `data/processed/train.csv` and evaluates on `data/processed/val.csv`
- fine-tunes a transformer classifier (`distilbert-base-uncased` by default)
- saves the best checkpoint to:
- `checkpoints/best_model.pt`

### Evaluation

Run:
```bash
python -m src.eval
```

This prints:

- overall accuracy
- per-class precision/recall/F1 (classification report)
- confusion matrix (rows=true, cols=pred)

and writes artifacts to:
- `reports/confusion_matrix_val.png`
- `reports/val_predictions.csv`

### Inference

You can run quick predictions in Python:
```bash
from src.inference import load_model_and_tokenizer, predict

tokenizer, model = load_model_and_tokenizer()  # loads checkpoints/best_model.pt by default

texts = [
    "I love this!",
    "You are stupid.",
    "All <group> should disappear."
]

results = predict(texts, tokenizer, model)
for r in results:
    print(r["label_name"], r["probs"])
```

### Results and known limitations

Overall validation accuracy can look strong while still failing on important minority classes.

In particular, the `hate` class can be heavily under-represented relative to `safe` and `toxic`. A common failure mode observed in this project is class collapse, where the model rarely or never predicts `hate` and instead routes those examples into `toxic` or `safe`. This can produce deceptively high overall accuracy while yielding near-zero recall for identity-based hate.

This behavior is inspected and documented using:
- `reports/val_predictions.csv`
- `notebooks/failure_analysis.ipynb`

Future work
- Improve minority-class recall with:
    - class-weighted loss,
    - oversampling,
    - threshold tuning / calibration,
    - alternative formulations (e.g., safe vs harmful binary classification).
- Add a simple real-time UI (Gradio) for interactive testing.
- Extend beyond text to multimodal safety signals (e.g., images) as a follow-up.

### Repository structure
- `src/`: model, datasets, tokenization, training, evaluation, inference
- `data/`: raw and processed data (typically not tracked in git)
- `notebooks/`: EDA and failure analysis
- `tests/`: basic tests
- `reports/`: evaluation artifacts (confusion matrix, predictions)
- `checkpoints/`: saved model weights