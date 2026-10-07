"""Interactive demo: detect emotions in photos or a live webcam, and compare model versions.

    python app.py      ->  open http://127.0.0.1:7860
"""
import html
import json
from functools import lru_cache

import gradio as gr
import pandas as pd

from src import model_info
from src.predictor import EMOJI, EMOTIONS, ROOT, EmotionPredictor, available_models

MODELS = available_models()
DEFAULT_MODEL = max(MODELS, key=lambda v: model_info.load(v).test_ferplus or 0)  # best on FER+ test
GITHUB = "https://github.com/tegemenozyurek/emotionDetecter"
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # same order as the matplotlib charts
MODEL_COLORS = {v: SERIES_COLORS[i % len(SERIES_COLORS)] for i, v in enumerate(MODELS)}


# ---------------------------------------------------------------- inference

@lru_cache(maxsize=None)
def get_predictor(version):
    return EmotionPredictor(version)


def analyze(image, version):
    if image is None:
        return None, None, ""
    predictor = get_predictor(version)
    faces, fallback = predictor.predict(image)
    faces.sort(key=lambda f: f.box[2] * f.box[3], reverse=True)  # largest face first
    main = faces[0]
    probs = {f"{EMOJI[e]} {e}": float(p) for e, p in zip(EMOTIONS, main.probs)}

    if fallback:
        info = "No face detected, so the whole image was classified as one face."
    elif len(faces) == 1:
        info = f"1 face detected"
    else:
        info = f"{len(faces)} faces detected · chart shows the largest"
    headline = (f"<div class='verdict'><span class='verdict-emoji'>{EMOJI[main.emotion]}</span>"
                f"<div><div class='verdict-label'>{main.emotion}</div>"
                f"<div class='verdict-sub'>{main.confidence:.0%} confident · {info} · {version}</div></div></div>")
    return predictor.annotate(image, faces), probs, headline


def analyze_stream(frame, version):
    if frame is None:
        return None
    predictor = get_predictor(version)
    faces, fallback = predictor.predict(frame)
    return frame if fallback else predictor.annotate(frame, faces)


# ---------------------------------------------------------------- about models

def pct(x):
    return "—" if x is None else f"{x:.1%}"


def model_cards():
    infos = [model_info.load(v) for v in MODELS]
    best = max((i.test_ferplus or 0) for i in infos)
    cards = []
    for i in infos:
        status = (f"<span class='pill pill-live'>training {i.epochs_done}/{i.config.get('epochs')}</span>"
                  if i.training else "")
        top = "<span class='pill pill-best'>best</span>" if i.test_ferplus and i.test_ferplus == best else ""
        cards.append(f"""
        <div class='card' data-model='{i.version}'>
          <div class='card-head'><span class='card-name'>{i.version}</span>{top}{status}</div>
          <div class='metrics'>
            <div><div class='metric'>{pct(i.test_ferplus)}</div><div class='metric-label'>FER+ test</div></div>
            <div><div class='metric'>{pct(i.distorted)}</div><div class='metric-label'>distorted</div></div>
            <div><div class='metric'>{pct(i.best_val)}</div><div class='metric-label'>best val</div></div>
          </div>
        </div>""")
    return f"<div class='cards'>{''.join(cards)}</div>"


def comparison_table():
    infos = [model_info.load(v) for v in MODELS]
    rows = [
        ("Epochs", lambda i: f"{i.epochs_done}/{i.config.get('epochs', '?')}"),
        ("Best validation", lambda i: pct(i.best_val)),
        ("Test · original labels", lambda i: pct(i.test_original)),
        ("Test · FER+ labels", lambda i: pct(i.test_ferplus)),
        ("Test · distorted faces", lambda i: pct(i.distorted)),
    ]
    head = "".join(f"<th data-model='{i.version}'>{i.version}</th>" for i in infos)
    body = "".join(
        f"<tr><td>{name}</td>{''.join(f'<td>{fn(i)}</td>' for i in infos)}</tr>" for name, fn in rows)
    return f"<table class='cmp'><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>"


def history_frame():
    rows = []
    for v in MODELS:
        for epoch, train_acc, val_acc in model_info.load(v).history:
            rows.append({"epoch": epoch, "validation accuracy (%)": round(val_acc * 100, 1), "model": v})
    return pd.DataFrame(rows)


def tooltips_html():
    tips = {v: model_info.load(v).tooltip() for v in MODELS}
    return f"<div id='model-tips' data-tips='{html.escape(json.dumps(tips), quote=True)}'></div>"


def image_path(version, kind):
    """Newest plot of this kind for a model (FER+ version preferred), or None."""
    for name in (f"{kind}_ferplus.png", f"{kind}.png"):
        path = ROOT / "assets" / version / name
        if path.exists():
            return str(path)
    return None


def robustness_path():
    path = ROOT / "assets" / "robustness_ferplus.png"
    return str(path) if path.exists() else str(ROOT / "assets" / "robustness.png")


def refresh():
    return (model_cards(), comparison_table(), history_frame(), tooltips_html(), robustness_path(),
            *[image_path(v, "training_curves") for v in MODELS],
            *[image_path(v, "confusion_matrix") for v in MODELS])


# ---------------------------------------------------------------- styling

THEME = gr.themes.Base(
    primary_hue="violet", secondary_hue="indigo", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    radius_size="lg",
).set(
    button_primary_background_fill="linear-gradient(90deg, *primary_600, *secondary_500)",
    button_primary_background_fill_hover="linear-gradient(90deg, *primary_500, *secondary_400)",
    button_primary_text_color="white",
    block_shadow="0 1px 2px rgb(0 0 0 / 0.04), 0 4px 16px rgb(0 0 0 / 0.04)",
)

CSS = """
.gradio-container { width: 100% !important; max-width: 1100px !important; margin: 0 auto !important; }
.gradio-container .main, .gradio-container .tabs, .gradio-container .tabitem { width: 100% !important; }
.hero { padding: 28px 4px 4px; display: flex; align-items: flex-end; gap: 16px; }
.hero h1 { font-size: 2.6rem; font-weight: 800; letter-spacing: -0.03em; margin: 0; line-height: 1.1; }
.hero h1 span { background: linear-gradient(90deg, #7c3aed, #6366f1 45%, #06b6d4);
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { margin: 6px 0 0; color: var(--body-text-color-subdued); font-size: 1.02rem; }
.hero a.gh { margin-left: auto; padding-bottom: 4px; font-size: .9rem; color: var(--body-text-color-subdued); text-decoration: none; }
.hero a.gh:hover { color: var(--color-accent); }

/* model picker: names only, details in a floating tooltip (see JS) */
#model-picker label { font-weight: 600; font-family: var(--font-mono); }
#tip-pop { position: fixed; z-index: 10000; max-width: 340px; white-space: pre-line; pointer-events: none;
  font-size: 12.5px; line-height: 1.55; padding: 12px 14px; border-radius: 12px;
  color: #f8fafc; background: #0f172a; border: 1px solid rgb(148 163 184 / .25);
  box-shadow: 0 12px 32px rgb(0 0 0 / .3); opacity: 0; transform: translateY(-4px);
  transition: opacity .12s, transform .12s; }
#tip-pop.show { opacity: 1; transform: translateY(0); }
#tip-pop b { display: block; font-size: 13.5px; margin-bottom: 4px; }

.verdict { display: flex; align-items: center; gap: 14px; padding: 14px 16px; border-radius: 14px;
  background: linear-gradient(135deg, rgb(124 58 237 / .10), rgb(6 182 212 / .08));
  border: 1px solid var(--border-color-primary); }
.verdict-emoji { font-size: 2.4rem; line-height: 1; }
.verdict-label { font-size: 1.5rem; font-weight: 700; text-transform: capitalize; }
.verdict-sub { color: var(--body-text-color-subdued); font-size: .9rem; }

.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
.card { padding: 18px; cursor: default; border-radius: 16px; background: var(--block-background-fill);
  border: 1px solid var(--border-color-primary); transition: transform .15s, box-shadow .15s; }
.card:hover { transform: translateY(-2px); box-shadow: 0 10px 28px rgb(0 0 0 / .08); }
.card-head { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.card-name { font-family: var(--font-mono); font-size: 1.35rem; font-weight: 700; margin-right: 4px; }
.pill { font-size: .75rem; padding: 3px 9px; border-radius: 999px; font-weight: 600;
  background: rgb(124 58 237 / .12); color: #7c3aed; }
.pill-best { background: rgb(16 185 129 / .14); color: #059669; }
.pill-live { background: rgb(245 158 11 / .16); color: #b45309; }
.metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 14px;
  padding-top: 12px; border-top: 1px solid var(--border-color-primary); }
.metric { font-size: 1.3rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.metric-label { font-size: .72rem; color: var(--body-text-color-subdued); text-transform: uppercase; letter-spacing: .04em; }

.cmp { width: 100%; border-collapse: collapse; font-size: .92rem; border: none !important; }
.cmp th, .cmp td { border: none !important; background: transparent !important; }
.cmp tr { border: none !important; }
.cmp th, .cmp td { padding: 10px 12px; border-bottom: 1px solid var(--border-color-primary) !important; text-align: right; }
.cmp tbody tr:hover td { background: rgb(124 58 237 / .06) !important; }
.cmp th:first-child, .cmp td:first-child { text-align: left; color: var(--body-text-color-subdued); }
.cmp th { font-family: var(--font-mono); font-size: 1rem; cursor: default; }
.cmp td { font-variant-numeric: tabular-nums; }
.section-title { font-size: 1.15rem; font-weight: 700; margin: 18px 0 2px; }
.plot-img img { border-radius: 12px; }
footer { display: none !important; }
"""

# Shows each model's description (from the hidden #model-tips element) in a
# floating tooltip when hovering its name (picker, cards, table). The tooltip lives on <body>
# so no Gradio container can clip it.
JS = """
() => {
  const SEL = '#model-picker label, [data-model]';
  const pop = document.createElement('div');
  pop.id = 'tip-pop';
  document.body.appendChild(pop);
  const tips = () => JSON.parse(document.querySelector('#model-tips')?.dataset.tips || '{}');
  document.addEventListener('mouseover', e => {
    const label = e.target.closest(SEL);
    if (!label) return;
    const text = tips()[label.dataset.model || label.querySelector('input')?.value];
    if (!text) return;
    const [title, ...rest] = text.split('\\n');
    pop.innerHTML = '';
    const b = document.createElement('b'); b.textContent = title;
    pop.append(b, document.createTextNode(rest.join('\\n')));
    const r = label.getBoundingClientRect();
    pop.style.left = Math.min(r.left, window.innerWidth - 360) + 'px';
    pop.style.top = (r.bottom + 10) + 'px';
    pop.classList.add('show');
  });
  document.addEventListener('mouseout', e => {
    if (e.target.closest(SEL) && !e.relatedTarget?.closest?.(SEL)) {
      pop.classList.remove('show');
    }
  });
}
"""

HERO = f"""
<div class='hero'>
  <div>
    <h1>emotion<span>Detecter</span></h1>
    <p>Facial emotion recognition · CNN trained from scratch</p>
  </div>
  <a class='gh' href='{GITHUB}' target='_blank'>GitHub ↗</a>
</div>
"""

# ---------------------------------------------------------------- layout

with gr.Blocks(title="emotionDetecter") as demo:
    gr.HTML(HERO)
    tips_holder = gr.HTML(tooltips_html(), visible="hidden")

    with gr.Tabs():
        with gr.Tab("🔍 Detect"):
            model_choice = gr.Radio(MODELS, value=DEFAULT_MODEL, label="Model", elem_id="model-picker")
            with gr.Tabs():
                with gr.Tab("📷 Photo"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1):
                            inp = gr.Image(label="Your photo", sources=["upload", "webcam", "clipboard"],
                                           type="numpy", height=420)
                            btn = gr.Button("Detect emotion", variant="primary", size="lg")
                        with gr.Column(scale=1):
                            out_img = gr.Image(label="Result", interactive=False, height=420)
                            verdict = gr.HTML()
                            out_probs = gr.Label(label="Probabilities", num_top_classes=7, show_label=False)
                    outputs = [out_img, out_probs, verdict]
                    btn.click(analyze, [inp, model_choice], outputs, api_name="analyze")
                    inp.change(analyze, [inp, model_choice], outputs, api_name=False)
                    model_choice.change(analyze, [inp, model_choice], outputs, api_name=False)

                with gr.Tab("🎥 Live webcam"):
                    with gr.Row():
                        cam = gr.Image(label="Webcam", sources=["webcam"], streaming=True, type="numpy")
                        live = gr.Image(label="Detected emotions", interactive=False)
                    cam.stream(analyze_stream, [cam, model_choice], live, stream_every=0.15)

        with gr.Tab("📊 About models"):
            with gr.Row():
                gr.HTML("<div class='section-title'>Models</div>")
                refresh_btn = gr.Button("↻ Refresh", size="sm", scale=0, min_width=110)
            cards = gr.HTML(model_cards())

            gr.HTML("<div class='section-title'>Comparison</div>")
            table = gr.HTML(comparison_table())

            gr.HTML("<div class='section-title'>Validation accuracy</div>")
            curve = gr.LinePlot(history_frame(), x="epoch", y="validation accuracy (%)", color="model",
                                color_map=MODEL_COLORS, height=340, show_label=False, y_lim=[0, 100])

            gr.HTML("<div class='section-title'>Robustness</div>")
            robust_img = gr.Image(robustness_path(), show_label=False, interactive=False,
                                  buttons=["fullscreen"], elem_classes="plot-img")

            gr.HTML("<div class='section-title'>FER+ label changes</div>")
            gr.Image(str(ROOT / "assets" / "ferplus" / "label_changes.png"), show_label=False, height=520,
                     interactive=False, buttons=["fullscreen"], elem_classes="plot-img")

            gr.HTML("<div class='section-title'>Training curves</div>")
            curve_imgs = [gr.Image(image_path(v, "training_curves"), label=v, interactive=False,
                                   buttons=["fullscreen"], elem_classes="plot-img") for v in MODELS]

            gr.HTML("<div class='section-title'>Confusion matrices</div>")
            with gr.Row():
                cm_imgs = [gr.Image(image_path(v, "confusion_matrix"), label=v, interactive=False,
                                    buttons=["fullscreen"], elem_classes="plot-img") for v in MODELS]

            refresh_btn.click(refresh, None, [cards, table, curve, tips_holder, robust_img, *curve_imgs, *cm_imgs],
                              api_name=False)

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, js=JS, allowed_paths=[str(ROOT / "assets")])
