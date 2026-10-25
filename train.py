"""
Assignment 2: training pipeline.

Downloads the PCB defect dataset (v3, 2x2 tiles), finetunes yolo11n, evaluates it
on the test split against two baselines (COCO yolo11n zero-shot, predict nothing),
scores every test tile, sweeps the tile-level threshold and writes everything the
app needs to results/. The trained model goes to weights/best.pt.

    uv run python main.py train
    uv run python train.py --sweep-only   # after changing the cost parameters, no retraining
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key

# --- Cost parameters (set these yourself, see REPORT.md) -----------------------
# Unit = one tile (a quarter of a board). Positive = the tile contains any defect.
COST_FP = None  # EUR, clean tile flagged as defective
COST_FN = None  # EUR, defective tile passed
COST_TP = None  # EUR, defective tile flagged
COST_TN = None  # EUR, clean tile passed
BASE_RATE = None  # share of production tiles that contain a defect (0-1)

# --- Data ----------------------------------------------------------------------
WORKSPACE = "srtimagedefectdetection-wuadb"
PROJECT = "pcb-defect-detection-hdqc8"
VERSION = 3
CLASS_NAMES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
SLIVER_SIDE = 0.01  # boxes with a side under 1% of the tile are mostly cut off by tiling

# --- Training ------------------------------------------------------------------
BASE_MODEL = "yolo11n.pt"
IMGSZ = 640
EPOCHS = 100
PATIENCE = 20
SEED = 0
CM_CONF = 0.25  # confidence for the box-level confusion matrix
THRESHOLDS = np.round(np.arange(0.0, 1.0001, 0.01), 2)

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets" / f"pcb-v{VERSION}"
RUNS_DIR = ROOT / "runs"
PRETRAINED_DIR = RUNS_DIR / "pretrained"
WEIGHTS_PATH = ROOT / "weights" / "best.pt"
RESULTS_DIR = ROOT / "results"
RUN_NAME = f"pcb-v{VERSION}-yolo11n"


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def download_dataset() -> Path:
    # Download once into datasets/ (gitignored); reruns reuse it.
    if not (DATASET_DIR / "data.yaml").exists():
        rf = Roboflow(api_key=load_api_key())
        rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
            "yolov11", location=str(DATASET_DIR)
        )
    # Roboflow writes relative paths and numeric class names; make both unambiguous.
    data = yaml.safe_load((DATASET_DIR / "data.yaml").read_text())
    data.update(
        path=str(DATASET_DIR),
        train=str(DATASET_DIR / "train" / "images"),
        val=str(DATASET_DIR / "valid" / "images"),
        test=str(DATASET_DIR / "test" / "images"),
        nc=len(CLASS_NAMES),
        names=dict(enumerate(CLASS_NAMES)),
    )
    (DATASET_DIR / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return DATASET_DIR / "data.yaml"


def read_labels(label_file: Path) -> list[tuple[int, float, float]]:
    """(class, width, height) per box, normalised to the tile."""
    if not label_file.exists():
        return []
    rows = [line.split() for line in label_file.read_text().splitlines() if line.strip()]
    return [(int(r[0]), float(r[3]), float(r[4])) for r in rows]


def is_sliver(box: tuple[int, float, float]) -> bool:
    return min(box[1], box[2]) < SLIVER_SIDE


def build_no_sliver_test(data_yaml: Path) -> Path:
    # Copy of the test split without sliver boxes, so box metrics can be reported both ways.
    src = DATASET_DIR / "test"
    dst = DATASET_DIR / "test_no_slivers"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src / "images", dst / "images")
    (dst / "labels").mkdir(parents=True)
    for label_file in (src / "labels").glob("*.txt"):
        kept = [
            line for line in label_file.read_text().splitlines()
            if line.strip() and min(float(line.split()[3]), float(line.split()[4])) >= SLIVER_SIDE
        ]
        (dst / "labels" / label_file.name).write_text("\n".join(kept) + ("\n" if kept else ""))
    data = yaml.safe_load(data_yaml.read_text())
    data["test"] = str(dst / "images")
    out = DATASET_DIR / "data_no_slivers.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    return out


def train(data_yaml: Path, device: str) -> Path:
    # Path under runs/ keeps the pretrained checkpoint out of the repo root.
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(PRETRAINED_DIR / BASE_MODEL))
    model.train(
        data=str(data_yaml),
        imgsz=IMGSZ,
        epochs=EPOCHS,
        patience=PATIENCE,
        seed=SEED,
        deterministic=True,
        device=device,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
    )
    # Only the finetuned model is committed; the app never reads runs/.
    WEIGHTS_PATH.parent.mkdir(exist_ok=True)
    shutil.copy2(Path(model.trainer.best), WEIGHTS_PATH)
    return WEIGHTS_PATH


def box_metrics(weights: Path, data_yaml: Path, device: str, name: str) -> dict:
    """Detection metrics on the test split: overall, per class, and the confusion matrix."""
    model = YOLO(str(weights))
    common = dict(data=str(data_yaml), split="test", imgsz=IMGSZ, device=device,
                  project=str(RUNS_DIR), exist_ok=True, verbose=False)
    m = model.val(name=f"{name}-test", **common)
    per_class = {
        CLASS_NAMES[c]: {
            "precision": float(m.box.p[i]),
            "recall": float(m.box.r[i]),
            "ap50": float(m.box.ap50[i]),
            "ap50_95": float(m.box.ap[i]),
        }
        for i, c in enumerate(m.box.ap_class_index)
    }
    # Separate pass because val's own matrix uses conf=0.001, which floods it with low-score boxes.
    cm = model.val(name=f"{name}-test-cm", conf=CM_CONF, plots=True, **common).confusion_matrix.matrix
    # Class-agnostic pass: same scoring as the COCO baseline, so the two are comparable.
    agnostic = model.val(name=f"{name}-test-agnostic", single_cls=True, **common)
    return {
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "per_class": per_class,
        "confusion_matrix": {
            "conf": CM_CONF,
            "labels": CLASS_NAMES + ["background"],  # rows = predicted, columns = true
            "matrix": cm.astype(int).tolist(),
        },
        "class_agnostic": {"map50": float(agnostic.box.map50), "map50_95": float(agnostic.box.map)},
    }


def coco_baseline(data_yaml: Path, device: str) -> dict:
    # COCO has no defect classes, so any box of any class counts as "defect here".
    model = YOLO(str(PRETRAINED_DIR / BASE_MODEL))
    m = model.val(data=str(data_yaml), split="test", imgsz=IMGSZ, device=device, single_cls=True,
                  project=str(RUNS_DIR), name="coco-zeroshot-test", exist_ok=True, verbose=False)
    return {"class_agnostic": {"map50": float(m.box.map50), "map50_95": float(m.box.map),
                               "precision": float(m.box.mp), "recall": float(m.box.mr)}}


def score_tiles(weights: Path, device: str) -> pd.DataFrame:
    """One row per test tile: ground truth plus max confidence overall and per class."""
    images = sorted((DATASET_DIR / "test" / "images").glob("*.jpg"))
    finetuned, coco = YOLO(str(weights)), YOLO(str(PRETRAINED_DIR / BASE_MODEL))
    # conf=0.001 keeps low-scoring boxes so the sweep can go all the way down.
    predict = dict(imgsz=IMGSZ, device=device, conf=0.001, stream=True, verbose=False)
    rows = []
    for img, res, coco_res in zip(images, finetuned.predict(images, **predict), coco.predict(images, **predict)):
        boxes = read_labels(DATASET_DIR / "test" / "labels" / f"{img.stem}.txt")
        solid = [b for b in boxes if not is_sliver(b)]
        source = img.stem.split(".rf.")[0]  # e.g. 01_missing_hole_02_jpg, shared by a board's 4 tiles
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        row = {
            "filename": img.name,
            "board_photo": source.removesuffix("_jpg"),
            "board_design": source.split("_")[0],
            "gt_defect": int(bool(boxes)),
            "gt_defect_no_slivers": int(bool(solid)),
            "only_slivers": int(bool(boxes) and not solid),
            "gt_classes": ";".join(sorted({CLASS_NAMES[b[0]] for b in boxes})),
            "gt_n_boxes": len(boxes),
            "max_conf": float(conf.max()) if len(conf) else 0.0,
            "coco_max_conf": float(coco_res.boxes.conf.max()) if len(coco_res.boxes) else 0.0,
        }
        for c, name in enumerate(CLASS_NAMES):
            row[f"max_conf_{name}"] = float(conf[cls == c].max()) if (cls == c).any() else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def costs_set() -> bool:
    return None not in (COST_FP, COST_FN, COST_TP, COST_TN, BASE_RATE)


def expected_cost_per_1000(tpr: float, fpr: float) -> float:
    # Reweights by the assumed production base rate, not the test split's defect share.
    defective = BASE_RATE * (tpr * COST_TP + (1 - tpr) * COST_FN)
    clean = (1 - BASE_RATE) * (fpr * COST_FP + (1 - fpr) * COST_TN)
    return 1000 * (defective + clean)


def sweep(tiles: pd.DataFrame) -> pd.DataFrame:
    """Tile-level confusion counts at each threshold, for each scorer and label variant."""
    variants = {
        ("finetuned", "all_labels"): (tiles, "gt_defect", "max_conf"),
        # Tiles whose only labels are slivers are ignored rather than called clean.
        ("finetuned", "no_slivers"): (tiles[tiles.only_slivers == 0], "gt_defect_no_slivers", "max_conf"),
        ("coco_zeroshot", "all_labels"): (tiles, "gt_defect", "coco_max_conf"),
        ("predict_nothing", "all_labels"): (tiles.assign(zero=0.0), "gt_defect", "zero"),
    }
    rows = []
    for (scorer, labels), (df, truth_col, score_col) in variants.items():
        truth = df[truth_col].to_numpy().astype(bool)
        score = df[score_col].to_numpy()
        for t in THRESHOLDS:
            # Predict-nothing never flags, regardless of threshold.
            flagged = score >= t if scorer != "predict_nothing" else np.zeros_like(truth)
            tp, fp = int((flagged & truth).sum()), int((flagged & ~truth).sum())
            fn, tn = int((~flagged & truth).sum()), int((~flagged & ~truth).sum())
            tpr, fpr = tp / max(tp + fn, 1), fp / max(fp + tn, 1)
            row = dict(scorer=scorer, labels=labels, threshold=t, tp=tp, fp=fp, fn=fn, tn=tn, tpr=tpr, fpr=fpr)
            if costs_set():
                row["cost_per_1000"] = expected_cost_per_1000(tpr, fpr)
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate(weights: Path, data_yaml: Path, device: str) -> tuple[dict, pd.DataFrame]:
    no_sliver_yaml = build_no_sliver_test(data_yaml)
    tiles = score_tiles(weights, device)
    metrics = {
        "eval_device": device,
        "finetuned": {
            "all_labels": box_metrics(weights, data_yaml, device, "finetuned"),
            "no_slivers": box_metrics(weights, no_sliver_yaml, device, "finetuned-no-slivers"),
        },
        "coco_zeroshot": coco_baseline(data_yaml, device),
        "predict_nothing": {"map50": 0.0, "map50_95": 0.0, "recall": 0.0},
        "test_tiles": {
            "total": len(tiles),
            "defective": int(tiles.gt_defect.sum()),
            "clean": int((tiles.gt_defect == 0).sum()),
            "only_slivers": int(tiles.only_slivers.sum()),
            "boards": int(tiles.board_photo.nunique()),
        },
    }
    return metrics, tiles


def save_sweep(metrics: dict, tiles: pd.DataFrame) -> None:
    """Sweep, pick the operating point from the current cost parameters, and write results/."""
    sweep_df = sweep(tiles)
    metrics["costs"] = {"fp": COST_FP, "fn": COST_FN, "tp": COST_TP, "tn": COST_TN, "base_rate": BASE_RATE}
    if costs_set():
        best = sweep_df.query("scorer == 'finetuned' and labels == 'all_labels'").nsmallest(1, "cost_per_1000")
        metrics["operating_point"] = best.iloc[0].to_dict()
    else:
        metrics["operating_point"] = None
        print("\nCost parameters at the top of train.py are not set; sweep saved without costs.")

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    sweep_df.to_csv(RESULTS_DIR / "threshold_sweep.csv", index=False)
    print("\nSaved results/metrics.json and results/threshold_sweep.csv")


def main() -> None:
    device = pick_device()
    data_yaml = download_dataset()
    weights = train(data_yaml, device)
    metrics, tiles = evaluate(weights, data_yaml, device)
    metrics["setup"] = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION},
        "base_model": BASE_MODEL, "imgsz": IMGSZ, "epochs": EPOCHS, "patience": PATIENCE,
        "seed": SEED, "train_device": device, "class_names": CLASS_NAMES, "sliver_side": SLIVER_SIDE,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    tiles.to_csv(RESULTS_DIR / "test_tiles.csv", index=False)
    save_sweep(metrics, tiles)


def sweep_only() -> None:
    # Cost changes only need the saved tile scores, not a new model.
    metrics = json.loads((RESULTS_DIR / "metrics.json").read_text())
    tiles = pd.read_csv(RESULTS_DIR / "test_tiles.csv")
    save_sweep(metrics, tiles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-only", action="store_true",
                        help="recompute the cost sweep from results/test_tiles.csv without training")
    if parser.parse_args().sweep_only:
        sweep_only()
    else:
        main()
