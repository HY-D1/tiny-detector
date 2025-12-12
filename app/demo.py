"""
Gradio demo for TinyDetector.

Run (inside Docker container):
    python app/demo.py
Then open:
    http://localhost:7860
"""

import torch
import gradio as gr

from src.inference import load_model_and_tokenizer, predict


def build_demo():
    # Load once at startup
    tokenizer, model = load_model_and_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def classify(text: str):
        text = (text or "").strip()
        if not text:
            return "Please enter some text.", {}

        result = predict([text], tokenizer, model, device=device)[0]

        label = result["label_name"]
        probs = result["probs"]  # dict: {safe: p, toxic: p, hate: p}

        # Return a nice label string + a probability dict for Gradio
        return label, probs

    with gr.Blocks(title="TinyDetector") as demo:
        gr.Markdown("# TinyDetector — Text Safety Classifier")
        gr.Markdown(
            "Enter text to classify it as **safe**, **toxic**, or **hate**. "
            "This demo loads the best saved checkpoint (`checkpoints/best_model.pt`)."
        )

        inp = gr.Textbox(
            lines=6,
            label="Input text",
            placeholder="Type or paste a comment here..."
        )

        btn = gr.Button("Classify")

        out_label = gr.Label(label="Prediction")
        out_probs = gr.JSON(label="Class probabilities")

        btn.click(fn=classify, inputs=inp, outputs=[out_label, out_probs])

        gr.Examples(
            examples=[
                ["I love this, thanks for sharing!"],
                ["This is the dumbest thing I've ever read."],
                ["Go back to where you came from."],
            ],
            inputs=inp,
            label="Examples",
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    # IMPORTANT for Docker: server_name="0.0.0.0"
    demo.launch(server_name="0.0.0.0", server_port=7860)
