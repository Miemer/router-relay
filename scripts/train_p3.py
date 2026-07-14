"""P3 LightGBM: train a 4-class tier classifier on labeled capture data.

Reads ``router-labeled-YYYY-MM-DD.jsonl`` (produced by ``realign_labels.py``),
extracts the 17 ``feature_snapshot`` scalars as input X and ``optimal_tier``
(c0..c3) as target Y, trains a LightGBM multiclass classifier, and saves the
model + feature order for inference (``ml_head.py``).

The model is a drop-in replacement for ``score_features`` at the
``runtime.apply_router`` call site — same FeatureBundle input, same ScoreResult
output, ~0.1ms inference.

Usage:
    uv run python scripts/train_p3.py --dates 2026-07-13,2026-07-14
    uv run python scripts/train_p3.py --dir logs --output models/p3_lightgbm.txt
    uv run python scripts/train_p3.py --auto          # auto-discover all dates
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from collections import Counter

import lightgbm as lgb
import numpy as np

# Feature order — must match features.py FeatureBundle.to_snapshot() exactly.
# This list is the stable contract between training and inference; it is saved
# alongside the model so ml_head.py can verify feature alignment at load time.
FEATURE_ORDER = [
    "char_len",
    "word_count",
    "zh_ratio",
    "code_ratio",
    "has_code_block",
    "has_json",
    "has_yaml",
    "has_table",
    "easy_kw_hits",
    "hard_kw_hits",
    "has_url",
    "has_file_ref",
    "n_messages",
    "total_context_chars",
    "turn_index",
    "complaint_detected",
    "complaint_hits",
]

TIERS = ("c0", "c1", "c2", "c3")
TIER_TO_IDX = {t: i for i, t in enumerate(TIERS)}

logger = logging.getLogger("train")


def _load_labeled(date: str, capture_dir: str) -> list[dict]:
    """Load one date's labeled records. Returns only records with a label."""
    path = os.path.join(capture_dir, f"router-labeled-{date}.jsonl")
    if not os.path.exists(path):
        logger.warning("missing labeled file: %s", path)
        return []
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("%s:%d: invalid JSON (%s)", path, lineno, exc)
                continue
            # Only keep records with a usable optimal_tier.
            if rec.get("optimal_tier") in TIER_TO_IDX:
                records.append(rec)
    return records


def _extract_xy(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and label vector Y from labeled records."""
    X_rows: list[list[float]] = []
    y_list: list[int] = []
    skipped = 0
    for rec in records:
        fs = rec.get("feature_snapshot") or {}
        if not isinstance(fs, dict):
            skipped += 1
            continue
        # Build feature row in canonical order; default missing fields to 0.
        row: list[float] = []
        ok = True
        for key in FEATURE_ORDER:
            val = fs.get(key, 0)
            if isinstance(val, bool):
                val = int(val)
            elif val is None:
                val = 0
            try:
                row.append(float(val))
            except (ValueError, TypeError):
                ok = False
                break
        if not ok:
            skipped += 1
            continue
        X_rows.append(row)
        y_list.append(TIER_TO_IDX[rec["optimal_tier"]])
    X = np.array(X_rows, dtype=np.float64) if X_rows else np.empty((0, len(FEATURE_ORDER)))
    y = np.array(y_list, dtype=np.int32) if y_list else np.empty(0, dtype=np.int32)
    return X, y


def _print_eval(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    """Print a compact classification report (confusion matrix + per-class stats)."""
    n = len(y_true)
    if n == 0:
        print(f"  {label}: (no data)")
        return
    acc = float(np.mean(y_true == y_pred))
    print(f"  {label}: accuracy={acc:.1%}  n={n}")

    # Confusion matrix.
    print(f"    {'':>8}  " + "  ".join(f"pred_{t:>2}" for t in TIERS))
    for i, tier in enumerate(TIERS):
        row = [int(np.sum((y_true == i) & (y_pred == j))) for j in range(len(TIERS))]
        row_total = sum(row)
        print(f"    true_{tier:>3}   " + "  ".join(f"{v:>8}" for v in row))

    # Per-class precision / recall.
    print("    per-class (precision / recall / support):")
    for i, tier in enumerate(TIERS):
        tp = int(np.sum((y_pred == i) & (y_true == i)))
        fp = int(np.sum((y_pred == i) & (y_true != i)))
        fn = int(np.sum((y_pred != i) & (y_true == i)))
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"      {tier}: {precision:.0%} / {recall:.0%} / {support}")


def train(
    dates: list[str],
    capture_dir: str,
    output: str,
    auto: bool,
    val_ratio: float,
    num_boost_round: int,
) -> dict:
    """Train a LightGBM 4-class tier classifier. Returns summary dict."""
    # Auto-discover all labeled dates if requested.
    if auto:
        pattern = os.path.join(capture_dir, "router-labeled-*.jsonl")
        found = [
            os.path.basename(p).replace("router-labeled-", "").replace(".jsonl", "")
            for p in sorted(glob.glob(pattern))
        ]
        dates = found
        logger.info("auto-discovered dates: %s", dates)

    if not dates:
        print("No dates specified. Use --dates or --auto.")
        return {"trained": False}

    # Load and merge all dates.
    all_records: list[dict] = []
    for date in dates:
        all_records.extend(_load_labeled(date, capture_dir))

    X, y = _extract_xy(all_records)
    n = len(y)
    tier_counts = Counter(y.tolist())

    print(f"\n{'='*60}")
    print(f"  P3 LightGBM Training")
    print(f"{'='*60}")
    print(f"  Dates:            {', '.join(dates)}")
    print(f"  Labeled records:  {n}")
    print(f"  Tier distribution: {dict((TIERS[k], v) for k, v in sorted(tier_counts.items()))}")

    if n < 20:
        print(f"\n  ⚠ Only {n} labeled records — need more data for meaningful training.")
        print(f"    Run judge_labels.py + realign_labels.py on more traffic first.")
        return {"trained": False, "n_samples": n}

    # Check minimum class diversity.
    n_classes = len([v for v in tier_counts.values() if v > 0])
    if n_classes < 2:
        print(f"\n  ⚠ Only {n_classes} tier class present — cannot train a classifier.")
        return {"trained": False, "n_samples": n, "n_classes": n_classes}

    # Train / validation split (stratified-ish: just random shuffle).
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"  Train / val:      {len(y_train)} / {len(y_val)}")
    print(f"  Features:         {len(FEATURE_ORDER)}")

    # LightGBM parameters for a small tabular multiclass problem.
    params = {
        "objective": "multiclass",
        "num_class": 4,
        "metric": "multi_logloss",
        "num_leaves": 15,
        "learning_rate": 0.1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "min_data_in_leaf": max(1, len(y_train) // 20),  # scale with data
    }

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=list(FEATURE_ORDER))
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, feature_name=list(FEATURE_ORDER))

    print(f"\n  Training LightGBM (num_boost_round={num_boost_round})...")
    model = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[lgb.log_evaluation(period=max(50, num_boost_round // 4))],
    )

    # Evaluate.
    y_pred_train = np.argmax(model.predict(X_train), axis=1)
    y_pred_val = np.argmax(model.predict(X_val), axis=1)

    print(f"\n{'='*60}")
    _print_eval(y_train, y_pred_train, "TRAIN")
    print()
    _print_eval(y_val, y_pred_val, "VAL  ")

    # Feature importance.
    imp = model.feature_importance(importance_type="gain")
    imp_pairs = sorted(zip(FEATURE_ORDER, imp), key=lambda x: -x[1])
    print(f"\n  Feature importance (gain):")
    for name, gain in imp_pairs[:8]:
        print(f"    {name:<24} {gain:>10.1f}")

    # Save model + metadata.
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    model.save_model(output)
    meta_path = output.replace(".txt", ".meta.json")
    if not meta_path.endswith(".json"):
        meta_path = output + ".meta.json"
    meta = {
        "feature_order": FEATURE_ORDER,
        "tiers": list(TIERS),
        "num_samples": n,
        "tier_distribution": dict((TIERS[k], v) for k, v in sorted(tier_counts.items())),
        "val_accuracy": float(np.mean(y_val == y_pred_val)),
        "schema_version": 1,
        "trained_on": dates,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Model saved: {output}")
    print(f"  Metadata:    {meta_path}")
    print(f"  To activate: set ROUTER_ML_MODEL_PATH={output} and restart server")
    print(f"{'='*60}")

    return {
        "trained": True,
        "n_samples": n,
        "val_accuracy": float(np.mean(y_val == y_pred_val)),
        "output": output,
    }


def _parse_args() -> argparse.Namespace:
    today_dates = ""  # require explicit --dates or --auto
    p = argparse.ArgumentParser(description="Train P3 LightGBM tier classifier.")
    p.add_argument("--dates", default=today_dates,
                   help="comma-separated YYYY-MM-DD dates (default: empty)")
    p.add_argument("--dir", dest="capture_dir", default="logs", help="capture directory")
    p.add_argument("--output", default="models/p3_lightgbm.txt", help="model output path")
    p.add_argument("--auto", action="store_true", help="auto-discover all labeled dates")
    p.add_argument("--val-ratio", type=float, default=0.2, help="validation split ratio")
    p.add_argument("--num-boost-round", type=int, default=200, help="boosting rounds")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()

    dates = [d.strip() for d in args.dates.split(",") if d.strip()] if args.dates else []
    summary = train(
        dates=dates,
        capture_dir=args.capture_dir,
        output=args.output,
        auto=args.auto,
        val_ratio=args.val_ratio,
        num_boost_round=args.num_boost_round,
    )
    return 0 if summary.get("trained") else 1


if __name__ == "__main__":
    raise SystemExit(main())
