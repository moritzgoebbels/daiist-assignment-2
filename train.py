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

import matplotlib

matplotlib.use("Agg")  # plots are written to files; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key

# --- Cost parameters (see REPORT.md) -------------------------------------------
# Unit = one tile (a quarter of a board). Positive = the tile contains any defect.
LABOUR_EUR_PER_HOUR = 45.0  # Eurostat, total employer labour cost, Germany
REVIEW_SECONDS_PER_TILE = 20
COST_FN = 50.0  # EUR, defect missed at fabrication, found at assembly
COST_AUTO_FAIL_FP = 5.0  # EUR, good 10x10 cm bare board scrapped (~15.5 in2 at ~$0.25/in2, plus handling)
COST_REVIEW = REVIEW_SECONDS_PER_TILE / 3600 * LABOUR_EUR_PER_HOUR  # EUR, every reviewed tile

# Single threshold = model alone: at or above it auto-fails, below it passes, no human.
COST_FP = COST_AUTO_FAIL_FP  # EUR, clean tile auto-failed
COST_TP = 0.0  # EUR, defective board is scrapped anyway
COST_TN = 0.0  # EUR, clean tile passed

# Board-level defect rate converted to tiles. Main case: a real defective board has one
# isolated defect, so one defective tile. Sensitivity: the test split's 203 defective
# tiles / 72 boards = 2.82, inflated by the dataset's several synthetic defects per photo.
BOARD_DEFECT_RATE = 0.03  # share of production boards with any defect
TILES_PER_BOARD = 4
DEFECTIVE_TILES_PER_DEFECTIVE_BOARD = 1
DEFECTIVE_TILES_PER_DEFECTIVE_BOARD_DATASET = 203 / 72
BASE_RATE = BOARD_DEFECT_RATE * DEFECTIVE_TILES_PER_DEFECTIVE_BOARD / TILES_PER_BOARD
BASE_RATE_DATASET = BOARD_DEFECT_RATE * DEFECTIVE_TILES_PER_DEFECTIVE_BOARD_DATASET / TILES_PER_BOARD

# Review band: score < lower passes, lower <= score < upper goes to a verifier,
# score >= upper fails automatically.
VERIFIER_ACCURACY = 0.95  # share of real defects the verifier catches; never rejects a clean tile
BOARDS_PER_HOUR = 30
HOURS_PER_DAY = 16
HOURS_PER_SHIFT = 8
TILES_PER_DAY = BOARDS_PER_HOUR * HOURS_PER_DAY * TILES_PER_BOARD  # 1920
N_BOOTSTRAP = 1000

# --- Data ----------------------------------------------------------------------
WORKSPACE = "srtimagedefectdetection-wuadb"
PROJECT = "pcb-defect-detection-hdqc8"
VERSION = 3
CLASS_NAMES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
SLIVER_SIDE = 0.01  # boxes with a side under 1% of the tile are mostly cut off by tiling
# Two "clean" test tiles of 12_spur_04 show a copper spur that the other photos of design 12 lack
# at the same spot, so they are likely unlabelled defects. Used only as a sensitivity case.
MISLABELLED_TILES = [
    "12_spur_04_jpg.rf.492e3a4397d49b1d7c8edd2621af2752.jpg",
    "12_spur_04_jpg.rf.4e76e88b05aeb7b252fa2354bda95f79.jpg",
]

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


def score_val_tiles(weights: Path, device: str) -> pd.DataFrame:
    """One row per validation tile, finetuned model only, for the validation check of the thresholds."""
    images = sorted((DATASET_DIR / "valid" / "images").glob("*.jpg"))
    predict = dict(imgsz=IMGSZ, device=device, conf=0.001, stream=True, verbose=False)
    rows = []
    for img, res in zip(images, YOLO(str(weights)).predict(images, **predict)):
        boxes = read_labels(DATASET_DIR / "valid" / "labels" / f"{img.stem}.txt")
        source = img.stem.split(".rf.")[0]
        conf = res.boxes.conf.cpu().numpy()
        rows.append({
            "filename": img.name,
            "board_photo": source.removesuffix("_jpg"),
            "board_design": source.split("_")[0],
            "gt_defect": int(bool(boxes)),
            "gt_classes": ";".join(sorted({CLASS_NAMES[b[0]] for b in boxes})),
            "max_conf": float(conf.max()) if len(conf) else 0.0,
        })
    return pd.DataFrame(rows)


def costs_set() -> bool:
    return None not in (COST_FP, COST_FN, COST_TP, COST_TN, BASE_RATE)


def expected_cost_per_1000(tpr: float, fpr: float, base_rate: float = None) -> float:
    # Reweights by the assumed production base rate, not the test split's defect share.
    base_rate = BASE_RATE if base_rate is None else base_rate
    defective = base_rate * (tpr * COST_TP + (1 - tpr) * COST_FN)
    clean = (1 - base_rate) * (fpr * COST_FP + (1 - fpr) * COST_TN)
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
                row["cost_per_1000_dataset_rate"] = expected_cost_per_1000(tpr, fpr, BASE_RATE_DATASET)
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


def band_params(**overrides) -> dict:
    """Main-case review band parameters; sensitivity cases override one of them."""
    params = dict(base_rate=BASE_RATE, cost_fn=COST_FN, cost_review=COST_REVIEW,
                  cost_auto_fail_fp=COST_AUTO_FAIL_FP, verifier_accuracy=VERIFIER_ACCURACY)
    params.update(overrides)
    return params


def tile_truth_and_score(tiles: pd.DataFrame, labels: str = "all_labels") -> tuple[np.ndarray, np.ndarray]:
    # Same label variants as the single-threshold sweep.
    if labels == "no_slivers":
        tiles = tiles[tiles.only_slivers == 0]
        return tiles.gt_defect_no_slivers.to_numpy().astype(bool), tiles.max_conf.to_numpy()
    if labels == "relabelled":
        truth = tiles.gt_defect.astype(bool) | tiles.filename.isin(MISLABELLED_TILES)
        return truth.to_numpy(), tiles.max_conf.to_numpy()
    return tiles.gt_defect.to_numpy().astype(bool), tiles.max_conf.to_numpy()


def share_below(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.sort(scores), thresholds, side="left") / len(scores)


def band_grid(truth: np.ndarray, score: np.ndarray, p: dict) -> np.ndarray:
    """Expected cost per 1000 tiles for every (lower, upper) pair; NaN where lower > upper."""
    pos_below, neg_below = share_below(score[truth], THRESHOLDS), share_below(score[~truth], THRESHOLDS)
    pos_pass, pos_upper = pos_below[:, None], pos_below[None, :]
    neg_pass, neg_upper = neg_below[:, None], neg_below[None, :]
    # A defect costs COST_FN if passed, or if reviewed and the verifier misses it.
    defective = (pos_pass * p["cost_fn"]
                 + (pos_upper - pos_pass) * (p["cost_review"] + (1 - p["verifier_accuracy"]) * p["cost_fn"]))
    clean = (neg_upper - neg_pass) * p["cost_review"] + (1 - neg_upper) * p["cost_auto_fail_fp"]
    cost = 1000 * (p["base_rate"] * defective + (1 - p["base_rate"]) * clean)
    cost[np.tril_indices(len(THRESHOLDS), k=-1)] = np.nan
    return cost


def band_optimum(truth: np.ndarray, score: np.ndarray, p: dict) -> tuple[float, float, float]:
    # Ties go to the first pair in grid order: lowest lower, then lowest upper.
    cost = band_grid(truth, score, p)
    i, j = np.unravel_index(np.nanargmin(cost), cost.shape)
    return float(THRESHOLDS[i]), float(THRESHOLDS[j]), float(cost[i, j])


def band_breakdown(truth: np.ndarray, score: np.ndarray, lower: float, upper: float, p: dict) -> dict:
    """Per 1000 production tiles: outcome x truth counts, TP/FP/FN/TN, cost and staffing."""
    acc, pi = p["verifier_accuracy"], p["base_rate"]
    pos, neg = score[truth], score[~truth]
    share = {
        "pass": ((pos < lower).mean(), (neg < lower).mean()),
        "review": (((pos >= lower) & (pos < upper)).mean(), ((neg >= lower) & (neg < upper)).mean()),
        "fail": ((pos >= upper).mean(), (neg >= upper).mean()),
    }
    unit_cost = {  # (defective, clean) cost per tile for each outcome
        "pass": (p["cost_fn"], 0.0),
        "review": (p["cost_review"] + (1 - acc) * p["cost_fn"], p["cost_review"]),
        "fail": (0.0, p["cost_auto_fail_fp"]),
    }
    outcomes = {}
    for outcome, (s_pos, s_neg) in share.items():
        n_pos, n_neg = 1000 * pi * s_pos, 1000 * (1 - pi) * s_neg
        outcomes[outcome] = {
            "tiles_defective": n_pos, "tiles_clean": n_neg,
            "cost": n_pos * unit_cost[outcome][0] + n_neg * unit_cost[outcome][1],
        }
    # Final decision after the verifier: a reviewed defect is caught with probability acc.
    reviewed_pos = outcomes["review"]["tiles_defective"]
    confusion = {
        "tp": outcomes["fail"]["tiles_defective"] + acc * reviewed_pos,
        "fn": outcomes["pass"]["tiles_defective"] + (1 - acc) * reviewed_pos,
        "fp": outcomes["fail"]["tiles_clean"],
        "tn": outcomes["pass"]["tiles_clean"] + outcomes["review"]["tiles_clean"],
    }
    confusion_cost = {
        "tp": acc * reviewed_pos * p["cost_review"],  # auto-failed defects cost nothing extra
        "fn": (outcomes["pass"]["tiles_defective"] * p["cost_fn"]
               + (1 - acc) * reviewed_pos * (p["cost_review"] + p["cost_fn"])),
        "fp": outcomes["fail"]["tiles_clean"] * p["cost_auto_fail_fp"],
        "tn": outcomes["review"]["tiles_clean"] * p["cost_review"],
    }
    review_share = (outcomes["review"]["tiles_defective"] + outcomes["review"]["tiles_clean"]) / 1000
    review_tiles_per_day = review_share * TILES_PER_DAY
    verifier_hours_per_day = review_tiles_per_day * REVIEW_SECONDS_PER_TILE / 3600
    shifts_per_day = HOURS_PER_DAY / HOURS_PER_SHIFT
    return {
        "lower": lower, "upper": upper,
        "cost_per_1000": sum(o["cost"] for o in outcomes.values()),
        "outcomes_per_1000": outcomes,
        "confusion_per_1000": confusion,
        "confusion_cost_per_1000": confusion_cost,
        "review_share": review_share,
        "review_tiles_per_day": review_tiles_per_day,
        "verifier_hours_per_day": verifier_hours_per_day,
        "verifiers_per_shift": verifier_hours_per_day / shifts_per_day / HOURS_PER_SHIFT,
    }


def sensitivity(tiles: pd.DataFrame, main_band: tuple[float, float], model_alone: float) -> pd.DataFrame:
    """Optimal band when one main-case parameter changes."""
    per_board = DEFECTIVE_TILES_PER_DEFECTIVE_BOARD / TILES_PER_BOARD
    cases = [
        ("main case", "", {}, "all_labels"),
        ("board defect rate 2%", "BOARD_DEFECT_RATE", {"base_rate": 0.02 * per_board}, "all_labels"),
        ("board defect rate 5%", "BOARD_DEFECT_RATE", {"base_rate": 0.05 * per_board}, "all_labels"),
        ("2.819 defective tiles per board (dataset)", "DEFECTIVE_TILES_PER_DEFECTIVE_BOARD",
         {"base_rate": BASE_RATE_DATASET}, "all_labels"),
        ("labour 13 EUR/h", "LABOUR_EUR_PER_HOUR",
         {"cost_review": REVIEW_SECONDS_PER_TILE / 3600 * 13}, "all_labels"),
        ("COST_FN 12 EUR", "COST_FN", {"cost_fn": 12.0}, "all_labels"),
        ("COST_FN 500 EUR", "COST_FN", {"cost_fn": 500.0}, "all_labels"),
        ("COST_AUTO_FAIL_FP 15 EUR", "COST_AUTO_FAIL_FP", {"cost_auto_fail_fp": 15.0}, "all_labels"),
        ("verifier accuracy 90%", "VERIFIER_ACCURACY", {"verifier_accuracy": 0.90}, "all_labels"),
        ("verifier accuracy 99%", "VERIFIER_ACCURACY", {"verifier_accuracy": 0.99}, "all_labels"),
        ("slivers ignored", "labels", {}, "no_slivers"),
        ("2 likely mislabelled tiles counted as defective", "MISLABELLED_TILES", {}, "relabelled"),
    ]
    rows = []
    for name, changed, overrides, labels in cases:
        truth, score = tile_truth_and_score(tiles, labels)
        p = band_params(**overrides)
        lower, upper, _ = band_optimum(truth, score, p)
        b = band_breakdown(truth, score, lower, upper, p)
        row = dict(case=name, changed=changed, lower=lower, upper=upper, cost_per_1000=b["cost_per_1000"],
                   review_share=b["review_share"], verifier_hours_per_day=b["verifier_hours_per_day"])
        if labels == "relabelled":
            # The main-case designs priced under the corrected labels.
            cost = lambda lo, up: band_breakdown(truth, score, lo, up, p)["cost_per_1000"]
            row.update(cost_model_alone=cost(model_alone, model_alone), cost_main_band=cost(*main_band),
                       cost_no_auto_fail=cost(main_band[0], 1.0))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_band(tiles: pd.DataFrame) -> dict:
    # Resampling test tiles shows how much the optimum rests on a handful of tiles (e.g. 2 false alarms).
    truth, score = tile_truth_and_score(tiles)
    rng = np.random.default_rng(SEED)
    p = band_params()
    draws = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(truth), len(truth))
        draws.append(band_optimum(truth[idx], score[idx], p))
    draws = np.array(draws)
    pct = lambda col: [float(np.percentile(draws[:, col], 5)), float(np.percentile(draws[:, col], 95))]
    return {"n": N_BOOTSTRAP, "lower_p5_p95": pct(0), "upper_p5_p95": pct(1), "cost_p5_p95": pct(2)}


# Parameter: (label, lowest and highest value searched, log-spaced search).
CROSSOVER_SEARCH = {
    "cost_fn": ("Missed defect cost (EUR)", 0.01, 10_000, True),
    "cost_auto_fail_fp": ("Scrapped good board cost (EUR)", 0.01, 10_000, True),
    "verifier_accuracy": ("Verifier accuracy", 0.0, 1.0, False),
    "cost_review": ("Review cost per tile (EUR)", 0.001, 1_000, True),
    "base_rate": ("Tile base rate", 1e-5, 0.99, True),
}


def first_band_change(truth, score, key: str, start: float, end: float, log: bool) -> tuple[float, tuple] | None:
    """Nearest value from start towards end where the chosen band changes, refined by bisection."""
    band = lambda v: band_optimum(truth, score, band_params(**{key: v}))[:2]
    base = band(start)
    steps = np.geomspace(start, end, 400) if log else np.linspace(start, end, 400)
    for same, changed in zip(steps[:-1], steps[1:]):
        if band(changed) != base:
            for _ in range(50):
                mid = np.sqrt(same * changed) if log else (same + changed) / 2
                same, changed = (mid, changed) if band(mid) == base else (same, mid)
            return float(changed), band(changed)
    return None


def crossover_note(key: str, value: float) -> str:
    # Translate a crossover back into the units the assumptions were stated in.
    if key == "base_rate":
        return f"board defect rate {value * TILES_PER_BOARD / DEFECTIVE_TILES_PER_DEFECTIVE_BOARD:.2%}"
    if key == "cost_review":
        return (f"labour EUR {value * 3600 / REVIEW_SECONDS_PER_TILE:.2f}/h at {REVIEW_SECONDS_PER_TILE} s, "
                f"or {value / LABOUR_EUR_PER_HOUR * 3600:.1f} s at EUR {LABOUR_EUR_PER_HOUR:.0f}/h")
    return ""


def crossovers(tiles: pd.DataFrame) -> pd.DataFrame:
    """For each assumption, how far it can move (down and up) before the chosen band changes."""
    truth, score = tile_truth_and_score(tiles)
    main = band_params()
    lower, upper, _ = band_optimum(truth, score, main)
    rows = []
    for key, (label, low, high, log) in CROSSOVER_SEARCH.items():
        for direction, end in (("down", low), ("up", high)):
            found = first_band_change(truth, score, key, main[key], end, log)
            rows.append({
                "parameter": key, "label": label, "main_value": main[key], "direction": direction,
                "searched_to": end, "crossover": found[0] if found else None,
                "band_before": f"{lower:.2f} / {upper:.2f}",
                "band_after": f"{found[1][0]:.2f} / {found[1][1]:.2f}" if found else None,
                "note": crossover_note(key, found[0]) if found else "no change in searched range",
            })
    return pd.DataFrame(rows)


def board_rollup(tiles: pd.DataFrame, lower: float, upper: float, verifier_accuracy: float) -> tuple[dict, pd.DataFrame]:
    """Board = its 4 tiles: fail if any tile fails, review if any is reviewed and none fails, else pass."""
    t = tiles.assign(decision=np.select([tiles.max_conf < lower, tiles.max_conf < upper], ["pass", "review"], "fail"))
    rows = []
    for board, g in t.groupby("board_photo"):
        defective = g.gt_defect == 1
        decision = "fail" if (g.decision == "fail").any() else "review" if (g.decision == "review").any() else "pass"
        reviewed_defects = int((defective & (g.decision == "review")).sum())
        # Caught if a defective tile auto-fails, else if the verifier spots one of its reviewed defective tiles.
        caught = 1.0 if (defective & (g.decision == "fail")).any() else 1 - (1 - verifier_accuracy) ** reviewed_defects
        rows.append({"board_photo": board, "board_score": g.max_conf.max(), "decision": decision,
                     "defective_tiles": int(defective.sum()), "reviewed_defective_tiles": reviewed_defects,
                     "p_caught": caught})
    boards = pd.DataFrame(rows)
    counts = boards.decision.value_counts()
    summary = {
        "lower": lower, "upper": upper, "boards": len(boards),
        "defective_boards": int((boards.defective_tiles > 0).sum()),
        "passed": int(counts.get("pass", 0)), "reviewed": int(counts.get("review", 0)), "failed": int(counts.get("fail", 0)),
        "recall_flagged": float((boards.decision != "pass").mean()),
        "recall_after_verifier": float(boards.p_caught.mean()),
        "note": "All test boards are defective, so no board-level false positive rate can be measured.",
    }
    return summary, boards


def validation_check(val_tiles: pd.DataFrame, tiles: pd.DataFrame, test_band: tuple[float, float]) -> pd.DataFrame:
    """Choose the thresholds on validation and price them on test, next to the test-chosen designs."""
    p = band_params()
    val, test = tile_truth_and_score(val_tiles), tile_truth_and_score(tiles)
    cost = lambda split, lo, up: band_breakdown(*split, lo, up, p)["cost_per_1000"]
    val_lower, val_upper, _ = band_optimum(*val, p)
    # Model alone is the lower = upper diagonal; ties go to the lowest threshold.
    val_alone = float(THRESHOLDS[np.nanargmin(np.diag(band_grid(*val, p)))])
    designs = [
        ("band chosen on validation", "validation", val_lower, val_upper),
        ("band chosen on test (main case)", "test", *test_band),
        ("no auto-fail (main lower, upper 1.0)", "test", test_band[0], 1.0),
        ("model alone chosen on validation", "validation", val_alone, val_alone),
    ]
    return pd.DataFrame([dict(design=name, chosen_on=chosen_on, lower=lo, upper=up,
                              cost_val=cost(val, lo, up), cost_test=cost(test, lo, up))
                         for name, chosen_on, lo, up in designs])


SERIES_BLUE, HIGHLIGHT, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"


def plot_cost_curve(sweep_df: pd.DataFrame, threshold: float, alternatives: dict) -> None:
    curve = sweep_df.query("scorer == 'finetuned' and labels == 'all_labels'")
    chosen = curve[curve.threshold == threshold].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    # Near 0 every clean tile is scrapped (thousands of EUR); cap the axis so the useful range stays readable.
    y_max = 1.5 * alternatives["do_nothing"]["cost_per_1000"]
    ax.plot(curve.threshold, curve.cost_per_1000, color=SERIES_BLUE, lw=2)
    for key, label, va in (("check_everything", "Person checks every tile", "top"), ("do_nothing", "Do nothing", "bottom")):
        y = alternatives[key]["cost_per_1000"]
        ax.axhline(y, color=MUTED, lw=1, ls="--")
        ax.text(0.99, y + (4 if va == "bottom" else -4), f"{label}: {y:.0f}", color=MUTED, ha="right", va=va, fontsize=9)
    ax.scatter([threshold], [chosen.cost_per_1000], s=64, color=HIGHLIGHT, edgecolor=SURFACE, lw=2, zorder=3)
    ax.annotate(f"chosen {threshold:.2f}: {chosen.cost_per_1000:.1f}", (threshold, chosen.cost_per_1000),
                xytext=(10, 10), textcoords="offset points", color=INK, fontsize=9)
    ax.text(0.07, y_max, f"curve clipped: {curve.cost_per_1000.max():,.0f} at threshold 0", color=MUTED,
            ha="left", va="top", fontsize=8, transform=ax.get_yaxis_transform())
    ax.set(xlabel="Confidence threshold (tile auto-fails if max score ≥ threshold)",
           ylabel="Expected cost, EUR per 1,000 tiles", xlim=(0, 1), ylim=(0, y_max))
    ax.set_title("Model alone: single-threshold cost (fine-tuned yolo11n, all labels)", color=INK, loc="left", fontsize=11)
    ax.grid(color=GRID, lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "cost_vs_threshold.png", dpi=150)
    plt.close(fig)


def plot_band_heatmap(grid: np.ndarray, lower: float, upper: float, cost: float) -> None:
    ramp = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
    ramp.set_bad(SURFACE)  # lower > upper is not a valid band
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    # Log scale: costs span roughly 40 to 400+, and the interesting region is the low end.
    img = ax.imshow(grid.T, origin="lower", extent=(-0.005, 1.005, -0.005, 1.005), cmap=ramp,
                    norm=LogNorm(vmin=np.nanmin(grid), vmax=np.nanmax(grid)), aspect="auto")
    ax.scatter([lower], [upper], s=80, color=HIGHLIGHT, edgecolor=SURFACE, lw=2, zorder=3)
    ax.annotate(f"chosen ({lower:.2f}, {upper:.2f}): {cost:.1f}", (lower, upper), xytext=(10, 8),
                textcoords="offset points", color=INK, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.85))
    bar = fig.colorbar(img, ax=ax)
    ticks = [v for v in (50, 100, 200, 500, 1000, 2000, 5000) if np.nanmin(grid) <= v <= np.nanmax(grid)]
    bar.set_ticks(ticks, labels=[f"{v:,}" for v in ticks])
    bar.set_label("Expected cost, EUR per 1,000 tiles (log scale)")
    ax.set(xlabel="Lower threshold (below: pass)", ylabel="Upper threshold (at or above: auto-fail)")
    ax.set_title("Review band cost over (lower, upper)", color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "band_heatmap.png", dpi=150)
    plt.close(fig)


def save_review_band(metrics: dict, tiles: pd.DataFrame, sweep_df: pd.DataFrame) -> None:
    """Pick the cost-minimising pass/review/fail band and price it against the alternatives."""
    truth, score = tile_truth_and_score(tiles)
    p = band_params()
    grid = band_grid(truth, score, p)
    lower, upper, cost = band_optimum(truth, score, p)
    near = np.argwhere(grid <= cost + 0.01)  # pairs tied with the optimum (within 1 cent)
    model_alone = metrics["operating_point"]["threshold"]
    metrics["review_band"] = {
        **band_breakdown(truth, score, lower, upper, p),
        "tied_lower_range": [float(THRESHOLDS[near[:, 0].min()]), float(THRESHOLDS[near[:, 0].max()])],
        "tied_upper_range": [float(THRESHOLDS[near[:, 1].min()]), float(THRESHOLDS[near[:, 1].max()])],
    }
    metrics["alternatives"] = {
        # lower = upper: nothing is reviewed, the model alone passes or fails.
        "model_alone": band_breakdown(truth, score, model_alone, model_alone, p),
        "check_everything": band_breakdown(truth, score, -np.inf, np.inf, p),
        "do_nothing": band_breakdown(truth, score, np.inf, np.inf, p),
    }
    sens = sensitivity(tiles, (lower, upper), model_alone)
    metrics["sensitivity"] = sens.to_dict(orient="records")
    metrics["bootstrap"] = bootstrap_band(tiles)
    cross = crossovers(tiles)
    metrics["crossovers"] = cross.to_dict(orient="records")
    board_summary, boards = board_rollup(tiles, lower, upper, VERIFIER_ACCURACY)
    metrics["board_rollup"] = board_summary
    metrics["costs"].update(cost_review=COST_REVIEW, cost_auto_fail_fp=COST_AUTO_FAIL_FP,
                            hours_per_day=HOURS_PER_DAY, hours_per_shift=HOURS_PER_SHIFT)

    sens.to_csv(RESULTS_DIR / "sensitivity.csv", index=False)
    cross.to_csv(RESULTS_DIR / "crossovers.csv", index=False)
    boards.to_csv(RESULTS_DIR / "board_rollup.csv", index=False)
    i_lo, i_up = np.meshgrid(THRESHOLDS, THRESHOLDS, indexing="ij")
    pd.DataFrame({"lower": i_lo.ravel(), "upper": i_up.ravel(), "cost_per_1000": grid.ravel()}).dropna().to_csv(
        RESULTS_DIR / "band_grid.csv", index=False)
    plot_cost_curve(sweep_df, model_alone, metrics["alternatives"])
    plot_band_heatmap(grid, lower, upper, cost)


def save_sweep(metrics: dict, tiles: pd.DataFrame, val_tiles: pd.DataFrame) -> None:
    """Sweep, pick the operating point from the current cost parameters, and write results/."""
    sweep_df = sweep(tiles)
    metrics["costs"] = {
        "fp": COST_FP, "fn": COST_FN, "tp": COST_TP, "tn": COST_TN, "base_rate": BASE_RATE,
        "board_defect_rate": BOARD_DEFECT_RATE,
        "defective_tiles_per_defective_board": DEFECTIVE_TILES_PER_DEFECTIVE_BOARD,
        "base_rate_dataset": BASE_RATE_DATASET,
        "defective_tiles_per_defective_board_dataset": DEFECTIVE_TILES_PER_DEFECTIVE_BOARD_DATASET,
        "labour_eur_per_hour": LABOUR_EUR_PER_HOUR, "review_seconds_per_tile": REVIEW_SECONDS_PER_TILE,
        "verifier_accuracy": VERIFIER_ACCURACY, "tiles_per_day": TILES_PER_DAY,
    }
    if costs_set():
        finetuned = sweep_df.query("scorer == 'finetuned' and labels == 'all_labels'")
        metrics["operating_point"] = finetuned.nsmallest(1, "cost_per_1000").iloc[0].to_dict()
        metrics["operating_point_dataset_rate"] = (
            finetuned.nsmallest(1, "cost_per_1000_dataset_rate").iloc[0].to_dict()
        )
        save_review_band(metrics, tiles, sweep_df)
        band = metrics["review_band"]
        check = validation_check(val_tiles, tiles, (band["lower"], band["upper"]))
        metrics["validation_check"] = check.to_dict(orient="records")
        check.to_csv(RESULTS_DIR / "validation_check.csv", index=False)
    else:
        metrics["operating_point"] = metrics["operating_point_dataset_rate"] = None
        print("\nCost parameters at the top of train.py are not set; sweep saved without costs.")

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    sweep_df.to_csv(RESULTS_DIR / "threshold_sweep.csv", index=False)
    print("\nSaved results/ (metrics.json, threshold_sweep.csv, review band, sensitivity, plots)")


def dataset_summary() -> dict:
    """What the report says about the dataset, computed from the downloaded labels."""
    version = Roboflow(api_key=load_api_key()).workspace(WORKSPACE).project(PROJECT).version(VERSION)
    summary = {"roboflow": {"preprocessing": version.preprocessing, "augmentation": version.augmentation}}
    for split in ("train", "valid", "test"):
        images = sorted((DATASET_DIR / split / "images").glob("*.jpg"))
        boxes = {img.name: read_labels(DATASET_DIR / split / "labels" / f"{img.stem}.txt") for img in images}
        all_boxes = [b for bs in boxes.values() for b in bs]
        sides = np.sqrt([w * h for _, w, h in all_boxes])  # box size as sqrt(area), relative to the tile side
        summary[split] = {
            "photos": len({img.stem.split(".rf.")[0] for img in images}),  # augmented copies share a source name
            "tiles": len(images),
            "defect_free_tiles": sum(not bs for bs in boxes.values()),
            "boxes": len(all_boxes),
            "boxes_per_class": {name: sum(b[0] == c for b in all_boxes) for c, name in enumerate(CLASS_NAMES)},
            "box_side_median": float(np.median(sides)),
            "box_side_median_px": float(np.median(sides) * IMGSZ),
            "sliver_boxes": sum(is_sliver(b) for b in all_boxes),
            "sliver_only_tiles": sum(bool(bs) and all(is_sliver(b) for b in bs) for bs in boxes.values()),
        }
    return summary


def training_summary() -> dict:
    """Epochs run, best epoch and wall time, from the Ultralytics log of the training run."""
    log = pd.read_csv(RUNS_DIR / RUN_NAME / "results.csv")
    log.columns = [c.strip() for c in log.columns]
    best = log.loc[log["metrics/mAP50-95(B)"].idxmax()]  # this Ultralytics version's fitness is mAP50-95
    return {
        "epochs_completed": int(log.epoch.max()),
        "best_epoch": int(best.epoch),
        "best_val_map50": float(best["metrics/mAP50(B)"]),
        "best_val_map50_95": float(best["metrics/mAP50-95(B)"]),
        "train_hours": float(log.time.iloc[-1] / 3600),
    }


def main() -> None:
    device = pick_device()
    data_yaml = download_dataset()
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "dataset_summary.json").write_text(json.dumps(dataset_summary(), indent=2))
    weights = train(data_yaml, device)
    metrics, tiles = evaluate(weights, data_yaml, device)
    metrics["training"] = training_summary()
    metrics["setup"] = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION},
        "base_model": BASE_MODEL, "imgsz": IMGSZ, "epochs": EPOCHS, "patience": PATIENCE,
        "seed": SEED, "train_device": device, "class_names": CLASS_NAMES, "sliver_side": SLIVER_SIDE,
    }
    tiles.to_csv(RESULTS_DIR / "test_tiles.csv", index=False)
    val_tiles = score_val_tiles(weights, device)
    val_tiles.to_csv(RESULTS_DIR / "val_tiles.csv", index=False)
    save_sweep(metrics, tiles, val_tiles)


def sweep_only() -> None:
    # Cost changes only need the saved tile scores, not a new model.
    metrics = json.loads((RESULTS_DIR / "metrics.json").read_text())
    tiles = pd.read_csv(RESULTS_DIR / "test_tiles.csv")
    val_tiles = pd.read_csv(RESULTS_DIR / "val_tiles.csv")
    save_sweep(metrics, tiles, val_tiles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-only", action="store_true",
                        help="recompute the cost sweep from results/test_tiles.csv without training")
    if parser.parse_args().sweep_only:
        sweep_only()
    else:
        main()
