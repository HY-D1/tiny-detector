from fastapi import FastAPI
from pydantic import BaseModel
import torch

from src.inference import load_model_and_tokenizer, predict


app = FastAPI(title="TinyDetector API")

@app.get("/")
def root():
    return {"status": "ok", "message": "TinyDetector API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

tokenizer, model = load_model_and_tokenizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()


class PredictRequest(BaseModel):
    texts: list[str]
    threshold: float = 0.0


@app.post("/predict")
def do_predict(req: PredictRequest):
    results = predict(req.texts, tokenizer, model, device=device, threshold=req.threshold)
    return {"results": results}
