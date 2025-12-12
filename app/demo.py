"""
Gradio demo for TinyDetector.

Run:
  python -m app.demo

Open:
  http://localhost:7860
"""

import torch
import gradio as gr

from src.inference import load_model_and_tokenizer, predict


def build_demo():
    tokenizer, model = load_model_and_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    def classify(text: str, threshold: float):
        text = (text or "").strip()
        if not text:
            return "Please enter some text.", {}

        result = predict([text], tokenizer, model, device=device, threshold=threshold)[0]
        label = f"{result['label_name']} (conf={result['confidence']:.3f})"
        return label, result["probs"]

    with gr.Blocks(title="TinyDetector") as demo:
        gr.Markdown("# TinyDetector — Text Safety Classifier")
        gr.Markdown(
            "Classifies text into **safe / toxic / hate** using the best saved checkpoint "
            "(`checkpoints/best_model.pt`)."
        )

        inp = gr.Textbox(lines=6, label="Input text", placeholder="Type or paste a comment here...")
        thr = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Uncertainty threshold (optional)")

        btn = gr.Button("Classify")
        out_label = gr.Label(label="Prediction")
        out_probs = gr.JSON(label="Class probabilities")

        btn.click(fn=classify, inputs=[inp, thr], outputs=[out_label, out_probs])

        gr.Examples(
            examples=[
                ["I love this, thanks for sharing!"],
                ["This is the dumbest thing I've ever read."],
                ["Go back to where you came from."],
                ["Genius idea… said no one ever."],
            ],
            inputs=inp,
            label="Examples",
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
