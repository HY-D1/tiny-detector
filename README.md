# TinyDetector: Lightweight Text Safety Classifier

TinyDetector is a small PyTorch-based text safety classifier that flags harmful content (e.g., toxicity and identity-based hate) using a fine-tuned transformer model. The repo includes data preparation, training, evaluation, inference utilities, and a simple demo/API for real-time testing.

## What it does

TinyDetector classifies text into one of three labels:

- `safe`
- `toxic`
- `hate`

Included components:

- Training pipeline (fine-tunes a transformer classifier; DistilBERT by default)
- Evaluation script (metrics + confusion matrix + per-example prediction export)
- Inference utilities (Python API + confidence thresholding)
- Gradio demo UI
- FastAPI inference service
- Notebooks for EDA and failure analysis
- Basic tests

## Dataset and label mapping

This project uses the **Jigsaw Toxic Comment Classification** dataset as the source of labeled comments. The original dataset contains multiple binary labels such as:

- `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

These are mapped into three coarse classes:

- `safe` (0): no toxic labels
- `hate` (2): `identity_hate == 1`
- `toxic` (1): any other toxic label (e.g., `toxic`, `insult`, `obscene`, `threat`, `severe_toxic`) without identity hate

A stratified train/validation split is saved to:

- `data/processed/train.csv`
- `data/processed/val.csv`

> Note: in most setups, raw/processed datasets are not tracked in git. If you keep them locally, make sure `data/` is ignored.

## Quickstart (Docker)

Build the dev image:

```bash
docker build -t tiny-detector:dev .
```

Start a container with the repo mounted:

```base
docker run --rm -it \
  -v "$(pwd)":/app \
  -p 7860:7860 \
  -p 8000:8000 \
  tiny-detector:dev \
  bash
```

Inside the container, install the package in editable mode (recommended):
```bash
pip install -e .
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
- macro F1
- per-class precision/recall/F1 (classification report)
- confusion matrix (rows=true, cols=pred)

and writes artifacts to:
- `reports/metrics.json`
- `reports/confusion_matrix_val.png`
- `reports/val_predictions.csv`

## Adjusting training and evaluation

Most knobs live in `src/config.py`.

Common changes:

### Train on full dataset (not the small debug subset)

In `src/config.py`:

- Set `max_train_samples = None` to use all training rows.
- Increase `num_epochs` if desired.

Example:

```python
max_train_samples = None
num_epochs = 3
train_batch_size = 16
learning_rate = 5e-5
```

### Change batch size / max token length
```python
train_batch_size = 16
eval_batch_size = 32
max_length = 128
```

If you hit out-of-memory (rare on CPU), reduce `train_batch_size` or `max_length`.

### Change the base model
```python
model_name = "distilbert-base-uncased"
```

You can swap to a smaller model for faster iteration (quality may drop), e.g.:
```python
model_name = "prajjwal1/bert-tiny"
```

### Imbalance handling (hate is rare)

You can try **one** of these at a time:
- class-weighted loss: `use_class_weighted_loss = True`
- weighted sampling: `use_weighted_sampler = True`

Avoid enabling both together at first, as it can destabilize training.

### Re-run evaluation after training
```bash
python -m src.eval
```

Evaluation artifacts are written to `reports/`:
- `reports/metrics.json`
- `reports/confusion_matrix_val.png`
- `reports/val_predictions.csv`
### Evaluate a specific checkpoint (and compare runs)

Default (uses `checkpoints/best_model.pt`):

```bash
python -m src.eval
```
Evaluate a specific checkpoint:
```bash
python -m src.eval --checkpoint checkpoints/best_model.pt --tag good
```

Compare two checkpoints:
```bash
python -m src.eval --checkpoint checkpoints/best_model.pt --tag good
python -m src.eval --checkpoint checkpoints/best_model_collapsed.pt --tag collapsed
```

This writes tagged outputs to `reports/`:
- `reports/metrics_<tag>.json`
- `reports/confusion_matrix_<tag>.png`
- `reports/val_predictions_<tag>.csv`

### Confidence thresholding

`predict(..., threshold=...)` enables a simple abstention mode:

- if the model's max probability is below `threshold`, the label is returned as `uncertain`

This is useful when treating the classifier as a safety filter where low-confidence cases should be reviewed.

### Demo (Gradio)

Start the interactive demo:
```basg
python -m app.demo
```

Open:

- http://localhost:7860

### API (FastAPI)

Start the API:
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Open:
- Swagger UI: http://localhost:8000/docs

Example request:
```bash
curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"texts":["I love this!", "You are stupid."], "threshold": 0.0}'
```

Testing
```bash
pytest -q
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
- Add image safety classification as a follow-up (multimodal extension)
- Add packaging improvements (e.g., versioned releases, pre-trained weights via GitHub Releases)

### Repository structure
- `app/`: Gradio demo + FastAPI service
- `src/`: model, datasets, tokenization, training, evaluation, inference
- `data/`: raw and processed data (often not tracked in git)
- `notebooks/`: EDA and failure analysis
- `tests/`: basic tests
- `reports/`: evaluation artifacts (metrics, confusion matrix, predictions)
- `checkpoints/`: saved model weights