"""
Assignment 2: Gradio dashboard for the PCB defect pass / review / fail decision.

Serves weights/best.pt on CPU and reads everything else from results/, which
train.py produced. Nothing is trained here. Cost logic is imported from
train.py so the dashboard and the pipeline can never disagree.

    uv run python main.py app
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.io.formats.style import Styler
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

import train

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TILE_SIZE = 640
TILE_NAMES = ["top-left", "top-right", "bottom-left", "bottom-right"]

# Dark-theme chart colours; decision colours are reserved for pass / review / fail only.
INK, MUTED, GRID = "#f0efec", "#898781", "#2c2c2a"
SERIES_BLUE, HIGHLIGHT, BAND_AQUA = "#3987e5", "#d95926", "#199e70"
DECISION_COLOURS = {"pass": "#0ca30c", "review": "#fab219", "fail": "#d03b3b"}
DECISION_TEXT = {"pass": "#ffffff", "review": "#0b0b0b", "fail": "#ffffff"}

# Everything below comes from the training run; the app only reads it.
metrics = json.loads((RESULTS / "metrics.json").read_text())
tiles = pd.read_csv(RESULTS / "test_tiles.csv")
sweep = pd.read_csv(RESULTS / "threshold_sweep.csv")
costs = metrics["costs"]
PARAMS = train.band_params(
    base_rate=costs["base_rate"], cost_fn=costs["fn"], cost_review=costs["cost_review"],
    cost_auto_fail_fp=costs["cost_auto_fail_fp"], verifier_accuracy=costs["verifier_accuracy"],
)
TRUTH, SCORE = train.tile_truth_and_score(tiles)
CHOSEN_LOWER, CHOSEN_UPPER = metrics["review_band"]["lower"], metrics["review_band"]["upper"]
MODEL_ALONE = metrics["operating_point"]["threshold"]
CHECK_ALL = train.band_breakdown(TRUTH, SCORE, float("-inf"), float("inf"), PARAMS)
DO_NOTHING = train.band_breakdown(TRUTH, SCORE, float("inf"), float("inf"), PARAMS)
model = YOLO(str(ROOT / "weights" / "best.pt"))


# --- Image inspection ------------------------------------------------------------

def decide(score: float, lower: float, upper: float) -> str:
    return "pass" if score < lower else "review" if score < upper else "fail"


def split_into_tiles(image: Image.Image) -> list[tuple[str, Image.Image]]:
    # Same preprocessing as dataset v3: 2x2 tiles, each stretched to 640x640.
    if image.size == (TILE_SIZE, TILE_SIZE):
        return [("tile", image)]
    w, h = image.size
    boxes = [(0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2), (0, h // 2, w // 2, h), (w // 2, h // 2, w, h)]
    return [(name, image.crop(box).resize((TILE_SIZE, TILE_SIZE))) for name, box in zip(TILE_NAMES, boxes)]


def predict(image: Image.Image | None) -> list | None:
    # Low conf keeps every candidate box; the sliders decide what is shown and how it is routed.
    if image is None:
        return None
    named = split_into_tiles(image.convert("RGB"))
    results = model.predict([t for _, t in named], imgsz=TILE_SIZE, conf=0.001, device="cpu", verbose=False)
    return [(name, r) for (name, _), r in zip(named, results)]


def tile_panel(tile: np.ndarray, label: str, decision: str) -> Image.Image:
    """Tile with a border and label bar in its decision colour."""
    border, bar = 10, 44
    panel = Image.new("RGB", (TILE_SIZE + 2 * border, TILE_SIZE + bar + border), DECISION_COLOURS[decision])
    panel.paste(Image.fromarray(tile), (border, bar))
    ImageDraw.Draw(panel).text((border + 4, 8), label, fill=DECISION_TEXT[decision],
                               font=ImageFont.load_default(size=26))
    return panel


def compose(panels: list[Image.Image]) -> Image.Image:
    # One image instead of a gallery, so the whole 2x2 grid always fits the frame.
    if len(panels) == 1:
        return panels[0]
    gap, (w, h) = 12, panels[0].size
    grid = Image.new("RGB", (2 * w + gap, 2 * h + gap), "#1a1a19")
    for i, panel in enumerate(panels):
        grid.paste(panel, ((i % 2) * (w + gap), (i // 2) * (h + gap)))
    return grid


def colour_decision(value: str) -> str:
    return f"color: {DECISION_COLOURS.get(value, INK)}; font-weight: 600"


def render(predictions: list | None, lower: float, upper: float):
    if not predictions:
        return None, pd.DataFrame()
    panels, rows = [], []
    for name, r in predictions:
        max_score = float(r.boxes.conf.max()) if len(r.boxes) else 0.0
        decision = decide(max_score, lower, upper)
        shown = r[r.boxes.conf >= lower]  # boxes below the lower threshold cannot change the decision
        panels.append(tile_panel(shown.plot()[..., ::-1], f"{name}: max {max_score:.3f} -> {decision.upper()}",
                                 decision))
        classes = sorted({shown.names[int(c)] for c in shown.boxes.cls})
        rows.append({"tile": name, "max score": max_score, "decision": decision,
                     "boxes shown": len(shown.boxes), "classes": ", ".join(classes)})
    table = pd.DataFrame(rows).style.map(colour_decision, subset=["decision"]).format({"max score": "{:.3f}"})
    return compose(panels), table


def inspect(image, lower, upper):
    predictions = predict(image)
    return (predictions, *render(predictions, lower, upper))


# --- Cost and review load ----------------------------------------------------------

def card(title: str, value: str, note: str) -> str:
    # Gradio theme variables keep the cards readable in light and dark mode.
    return (
        "<div style='padding:14px 16px;border:1px solid var(--border-color-primary);border-radius:8px;"
        "background:var(--block-background-fill)'>"
        f"<div style='font-size:0.85em;color:var(--body-text-color-subdued)'>{title}</div>"
        f"<div style='font-size:1.8em;font-weight:600;margin:2px 0'>{value}</div>"
        f"<div style='font-size:0.85em;color:var(--body-text-color-subdued)'>{note}</div></div>"
    )


def saving_note(band: float, check_all: float, fmt: str, unit: str = "") -> str:
    diff = check_all - band
    direction = "less" if diff >= 0 else "more"
    return f"{unit}{abs(diff):{fmt}} {direction} than a person checking every tile ({unit}{check_all:{fmt}})"


def summary_cards(b: dict) -> tuple[str, str, str]:
    return (
        card("Expected cost per 1,000 tiles", f"€{b['cost_per_1000']:.2f}",
             saving_note(b["cost_per_1000"], CHECK_ALL["cost_per_1000"], ".2f", "€")),
        card("Share sent to review", f"{b['review_share']:.1%}",
             f"{b['review_tiles_per_day']:.1f} tiles/day, {b['verifier_hours_per_day']:.2f} verifier h/day "
             f"(vs {CHECK_ALL['review_share']:.1%} when a person checks every tile)"),
        card(f"Verifiers per {costs['hours_per_shift']} h shift", f"{b['verifiers_per_shift']:.3f}",
             saving_note(b["verifiers_per_shift"], CHECK_ALL["verifiers_per_shift"], ".3f")),
    )


def routing_table(b: dict, lower: float, upper: float) -> Styler:
    """Where tiles go, as test-split counts and per 1,000 production tiles."""
    rows = []
    for outcome, o in b["outcomes_per_1000"].items():
        on_test = np.array([decide(s, lower, upper) == outcome for s in SCORE])
        rows.append({
            "Outcome": outcome,
            "Defective (test tiles)": int((TRUTH & on_test).sum()),
            "Clean (test tiles)": int((~TRUTH & on_test).sum()),
            "Defective per 1,000": round(o["tiles_defective"], 2),
            "Clean per 1,000": round(o["tiles_clean"], 2),
            "Cost € per 1,000": round(o["cost"], 2),
        })
    return pd.DataFrame(rows).style.map(colour_decision, subset=["Outcome"]).format(precision=2)


def confusion_table(b: dict) -> pd.DataFrame:
    """Final decision after the verifier, per 1,000 production tiles."""
    c, cost = b["confusion_per_1000"], b["confusion_cost_per_1000"]
    return pd.DataFrame([
        {"Actual": "defective", "Final: fail": f"TP {c['tp']:.2f}", "Final: pass": f"FN {c['fn']:.2f}",
         "Cost € per 1,000": round(cost["tp"] + cost["fn"], 2)},
        {"Actual": "clean", "Final: fail": f"FP {c['fp']:.2f}", "Final: pass": f"TN {c['tn']:.2f}",
         "Cost € per 1,000": round(cost["fp"] + cost["tn"], 2)},
    ])


def comparison_table(lower: float, upper: float) -> pd.DataFrame:
    designs = [
        (f"Band, current sliders ({lower:.2f} / {upper:.2f})", lower, upper),
        (f"Band, chosen ({CHOSEN_LOWER:.2f} / {CHOSEN_UPPER:.2f})", CHOSEN_LOWER, CHOSEN_UPPER),
        (f"Model alone ({MODEL_ALONE:.2f}, no review)", MODEL_ALONE, MODEL_ALONE),
        ("Person checks every tile", float("-inf"), float("inf")),
        ("Do nothing", float("inf"), float("inf")),
    ]
    rows = []
    for name, lo, up in designs:
        b = train.band_breakdown(TRUTH, SCORE, lo, up, PARAMS)
        rows.append({"Design": name, "Cost € per 1,000": round(b["cost_per_1000"], 2),
                     "Review share": f"{b['review_share']:.1%}",
                     "Review tiles/day": round(b["review_tiles_per_day"], 1),
                     "Verifier hours/day": round(b["verifier_hours_per_day"], 2)})
    return pd.DataFrame(rows)


def cost_curve(lower: float, upper: float, band_cost: float) -> go.Figure:
    curve = sweep.query("scorer == 'finetuned' and labels == 'all_labels'")
    chosen_cost = curve.loc[curve.threshold == MODEL_ALONE, "cost_per_1000"].iloc[0]
    fig = go.Figure()
    fig.add_vrect(x0=lower, x1=upper, fillcolor=BAND_AQUA, opacity=0.15, line_width=0,
                  annotation_text="review band", annotation_position="top left",
                  annotation_font_color=INK)
    fig.add_trace(go.Scatter(x=curve.threshold, y=curve.cost_per_1000, mode="lines", name="Model alone",
                             line=dict(color=SERIES_BLUE, width=2),
                             hovertemplate="threshold %{x:.2f}<br>€%{y:.2f} per 1,000 tiles<extra></extra>"))
    fig.add_trace(go.Scatter(x=[MODEL_ALONE], y=[chosen_cost], mode="markers",
                             name=f"Model alone, chosen {MODEL_ALONE:.2f}: €{chosen_cost:.2f}",
                             marker=dict(color=HIGHLIGHT, size=10, line=dict(color=INK, width=1.5))))
    fig.add_hline(y=band_cost, line=dict(color=BAND_AQUA, width=2),
                  annotation_text=f"Band ({lower:.2f} / {upper:.2f}): {band_cost:.2f}",
                  annotation_position="top right", annotation_font_color=INK)
    for y, label in ((CHECK_ALL["cost_per_1000"], "Person checks every tile"),
                     (DO_NOTHING["cost_per_1000"], "Do nothing")):
        fig.add_hline(y=y, line=dict(color=MUTED, width=1, dash="dash"),
                      annotation_text=f"{label}: {y:.2f}", annotation_position="top right",
                      annotation_font_color=MUTED)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK), height=440, margin=dict(l=60, r=20, t=20, b=40),
        xaxis=dict(title="Threshold (model alone: tile auto-fails if max score ≥ threshold)",
                   range=[0, 1], gridcolor=GRID, zeroline=False),
        yaxis=dict(title="Expected cost, € per 1,000 tiles", range=[0, 1.5 * DO_NOTHING["cost_per_1000"]],
                   gridcolor=GRID, zeroline=False),
        legend=dict(orientation="h", x=0, y=-0.18),  # below the plot, clear of the band label
    )
    return fig


def update_dashboard(lower: float, upper: float):
    note = ""
    if lower > upper:
        upper, note = lower, "*Lower is above upper, so upper is treated as equal to lower (no review band).*"
    b = train.band_breakdown(TRUTH, SCORE, lower, upper, PARAMS)
    return (*summary_cards(b), note, cost_curve(lower, upper, b["cost_per_1000"]),
            routing_table(b, lower, upper), confusion_table(b), comparison_table(lower, upper))


# --- Static model results ------------------------------------------------------------

def overall_table() -> pd.DataFrame:
    f = metrics["finetuned"]
    rows = [
        ("Fine-tuned yolo11n, all labels", f["all_labels"]["map50"], f["all_labels"]["map50_95"]),
        ("Fine-tuned yolo11n, slivers ignored", f["no_slivers"]["map50"], f["no_slivers"]["map50_95"]),
        ("Fine-tuned yolo11n, class-agnostic", f["all_labels"]["class_agnostic"]["map50"],
         f["all_labels"]["class_agnostic"]["map50_95"]),
        ("COCO yolo11n zero-shot, class-agnostic", metrics["coco_zeroshot"]["class_agnostic"]["map50"],
         metrics["coco_zeroshot"]["class_agnostic"]["map50_95"]),
        ("Predict nothing", metrics["predict_nothing"]["map50"], metrics["predict_nothing"]["map50_95"]),
    ]
    return pd.DataFrame(rows, columns=["Model (test split)", "mAP50", "mAP50-95"]).round(3)


def per_class_table() -> pd.DataFrame:
    all_labels = metrics["finetuned"]["all_labels"]["per_class"]
    no_slivers = metrics["finetuned"]["no_slivers"]["per_class"]
    return pd.DataFrame([
        {"Class": name, "AP50": r["ap50"], "AP50-95": r["ap50_95"], "Precision": r["precision"],
         "Recall": r["recall"], "AP50, slivers ignored": no_slivers[name]["ap50"]}
        for name, r in all_labels.items()
    ]).round(3)


def box_confusion_table() -> pd.DataFrame:
    cm = metrics["finetuned"]["all_labels"]["confusion_matrix"]
    return pd.DataFrame(cm["matrix"], columns=[f"true {n}" for n in cm["labels"]]).assign(
        predicted=cm["labels"]).set_index("predicted").reset_index()


CHANGED_NAMES = {
    "": "—",
    "BOARD_DEFECT_RATE": "Board defect rate",
    "DEFECTIVE_TILES_PER_DEFECTIVE_BOARD": "Defective tiles per defective board",
    "LABOUR_EUR_PER_HOUR": "Verifier labour cost",
    "COST_FN": "Cost of a missed defect",
    "COST_AUTO_FAIL_FP": "Cost of scrapping a good board",
    "VERIFIER_ACCURACY": "Verifier accuracy",
    "labels": "Test labels (slivers)",
}


CASE_NAMES = {
    "main case": "Main case",
    "labour 13 EUR/h": "Labour €13/h",
    "COST_FN 12 EUR": "Missed defect costs €12",
    "COST_FN 500 EUR": "Missed defect costs €500",
    "COST_AUTO_FAIL_FP 15 EUR": "Scrapped good board costs €15",
    "2.819 defective tiles per board (dataset)": "2.819 defective tiles per board (dataset)",
}


def sensitivity_table() -> pd.DataFrame:
    s = pd.DataFrame(metrics["sensitivity"])
    return pd.DataFrame({
        "Case": s.case.map(lambda c: CASE_NAMES.get(c, c[0].upper() + c[1:])),
        "Changed assumption": s.changed.map(CHANGED_NAMES),
        "Lower": s.lower.map("{:.2f}".format),
        "Upper": s.upper.map("{:.2f}".format),
        "Cost € per 1,000": s.cost_per_1000.map("{:.2f}".format),
        "Review share": s.review_share.map("{:.1%}".format),
        "Verifier hours/day": s.verifier_hours_per_day.map("{:.2f}".format),
    })


# --- Layout ----------------------------------------------------------------------------

with gr.Blocks(title="PCB defect review band") as demo:
    ds = metrics["setup"]["dataset"]
    gr.Markdown(
        "# PCB defect inspection: pass / review / fail\n"
        f"Fine-tuned yolo11n on Roboflow `{ds['workspace']}/{ds['project']}` v{ds['version']} (2x2 tiles). "
        f"Unit = one tile (a quarter of a board). Assumed production base rate **{costs['base_rate']:.2%}** of tiles "
        f"defective. Costs: missed defect €{costs['fn']:.0f}, review €{costs['cost_review']:.2f} per tile, "
        f"good board auto-failed €{costs['cost_auto_fail_fp']:.0f}, verifier catches "
        f"{costs['verifier_accuracy']:.0%} of defects. Rates come from the {len(tiles)} test tiles."
    )
    with gr.Row():
        lower = gr.Slider(0, 1, value=CHOSEN_LOWER, step=0.01, label="Lower threshold (below: pass)")
        upper = gr.Slider(0, 1, value=CHOSEN_UPPER, step=0.01, label="Upper threshold (at or above: auto-fail)")

    with gr.Tabs():
        with gr.Tab("Cost and review load"):
            with gr.Row():
                cost_card, review_card, staff_card = gr.HTML(), gr.HTML(), gr.HTML()
            band_note = gr.Markdown()
            curve = gr.Plot(label="Cost against threshold")
            gr.Markdown("**Where tiles go** (test-split counts, and per 1,000 production tiles at the assumed base rate)")
            routing = gr.Dataframe(interactive=False, wrap=True)
            gr.Markdown("**Final decision after the verifier** (per 1,000 production tiles)")
            confusion = gr.Dataframe(interactive=False, wrap=True)
            gr.Markdown("**Compared with the alternatives**")
            comparison = gr.Dataframe(interactive=False, wrap=True)

        with gr.Tab("Inspect an image"):
            gr.Markdown("Upload a 640x640 tile, or a full board photo (split into 2x2 tiles like the training data). "
                        "Boxes below the lower threshold are hidden.")
            with gr.Row():
                with gr.Column(scale=1):
                    image = gr.Image(type="pil", label="Tile or board photo", sources=["upload"], height=320)
                    tile_table = gr.Dataframe(interactive=False, wrap=True)
                tiles_view = gr.Image(type="pil", label="Tiles", interactive=False, height=640, scale=2)
            predictions = gr.State()

        with gr.Tab("Model performance"):
            gr.Markdown("**Overall** (test split, 288 tiles)")
            gr.Dataframe(overall_table(), interactive=False, wrap=True)
            gr.Markdown("**Per class** (precision and recall at the confidence that maximises mean F1)")
            gr.Dataframe(per_class_table(), interactive=False, wrap=True)
            gr.Markdown(f"**Box-level confusion matrix** (conf ≥ "
                        f"{metrics['finetuned']['all_labels']['confusion_matrix']['conf']}; background = missed / false box)")
            gr.Dataframe(box_confusion_table(), interactive=False, wrap=True)

        with gr.Tab("Sensitivity"):
            gr.Markdown("**Cost-minimising band when one assumption changes**")
            gr.Dataframe(sensitivity_table(), interactive=False, wrap=True)
            bs = metrics["bootstrap"]
            gr.Markdown(
                f"**Bootstrap** ({bs['n']} resamples of the test tiles), 5th-95th percentile: "
                f"lower {bs['lower_p5_p95'][0]:.2f}-{bs['lower_p5_p95'][1]:.2f}, "
                f"upper {bs['upper_p5_p95'][0]:.2f}-{bs['upper_p5_p95'][1]:.2f}, "
                f"cost €{bs['cost_p5_p95'][0]:.2f}-{bs['cost_p5_p95'][1]:.2f} per 1,000 tiles."
            )
            gr.Image(str(RESULTS / "band_heatmap.png"), label="Band cost over (lower, upper)", interactive=False)

    dashboard = [cost_card, review_card, staff_card, band_note, curve, routing, confusion, comparison]
    gr.on([lower.change, upper.change], update_dashboard, inputs=[lower, upper], outputs=dashboard)
    gr.on([lower.change, upper.change], render, inputs=[predictions, lower, upper], outputs=[tiles_view, tile_table])
    image.change(inspect, inputs=[image, lower, upper], outputs=[predictions, tiles_view, tile_table])
    demo.load(update_dashboard, inputs=[lower, upper], outputs=dashboard)


if __name__ == "__main__":
    demo.launch()
