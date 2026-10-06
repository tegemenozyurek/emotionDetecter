"""Interactive demo: pick a model, then upload a photo or use your webcam.

    python app.py      ->  open http://127.0.0.1:7860
"""
from functools import lru_cache

import gradio as gr

from src.predictor import EMOJI, EMOTIONS, EmotionPredictor, available_models

MODELS = available_models()
DESCRIPTIONS = {
    "eD0.1": "baseline CNN, no augmentation",
    "eD0.5": "+ data augmentation, longer training",
}


@lru_cache(maxsize=None)
def get_predictor(version):
    return EmotionPredictor(version)


def analyze(image, version):
    if image is None:
        return None, None, "Upload a photo or take one with your webcam."
    predictor = get_predictor(version)
    faces, fallback = predictor.predict(image)
    faces.sort(key=lambda f: f.box[2] * f.box[3], reverse=True)  # largest face first
    main = faces[0]
    probs = {f"{EMOJI[e]} {e}": float(p) for e, p in zip(EMOTIONS, main.probs)}

    if fallback:
        info = "No face detected — classified the whole image as one face."
    elif len(faces) == 1:
        info = f"1 face detected → **{main.emotion}** ({main.confidence:.0%})"
    else:
        found = ", ".join(f"{f.emotion} {f.confidence:.0%}" for f in faces)
        info = f"{len(faces)} faces detected: {found}. Chart shows the largest face."
    return predictor.annotate(image, faces), probs, f"`{version}` · {info}"


def analyze_stream(frame, version):
    if frame is None:
        return None
    predictor = get_predictor(version)
    faces, fallback = predictor.predict(frame)
    return frame if fallback else predictor.annotate(frame, faces)


with gr.Blocks(title="emotionDetecter") as demo:
    gr.Markdown(
        "# 😄 emotionDetecter\n"
        "Facial emotion recognition with a CNN trained from scratch on FER-2013 (7 emotions)."
    )
    model_choice = gr.Radio(
        choices=[(f"{v} — {DESCRIPTIONS.get(v, '')}".rstrip(" —"), v) for v in MODELS],
        value=MODELS[-1], label="Model",
    )
    with gr.Tab("📷 Photo"):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(label="Input", sources=["upload", "webcam", "clipboard"], type="numpy")
                btn = gr.Button("Detect emotion", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="Result", interactive=False)
                info = gr.Markdown()
                out_probs = gr.Label(label="Emotion probabilities", num_top_classes=7)
        outputs = [out_img, out_probs, info]
        btn.click(analyze, [inp, model_choice], outputs)
        inp.change(analyze, [inp, model_choice], outputs)
        model_choice.change(analyze, [inp, model_choice], outputs)  # re-run when switching models

    with gr.Tab("🎥 Live webcam"):
        gr.Markdown("Allow camera access and the boxes update in real time.")
        with gr.Row():
            cam = gr.Image(label="Webcam", sources=["webcam"], streaming=True, type="numpy")
            live = gr.Image(label="Detected emotions", interactive=False)
        cam.stream(analyze_stream, [cam, model_choice], live, stream_every=0.15)

if __name__ == "__main__":
    demo.launch()
