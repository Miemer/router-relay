"""P3 LightGBM: train a 4-class tier classifier on labeled capture data.

Reads ``router-labeled-YYYY-MM-DD.jsonl`` (produced by ``realign_labels.py``),
extracts the 17 ``feature_snapshot`` scalars as input X and ``optimal_tier``
(c0..c3) as target Y, trains a LightGBM multiclass classifier, and saves the
model + feature order for inference (``ml_head.py``).

Evaluation uses a **time-window holdout** (``--holdout-days``): the most recent
N days of labeled data are held out by time, and the new model is compared
against the currently-serving ``--active-model`` on that future window. This
mimics deployment ("does a model trained on the past generalize to the
future?") and is the gate the self-learning loop uses to avoid regressions —
unlike a random split, which leaks distribution and can hide them.

The model is a drop-in replacement for ``score_features`` at the
``runtime.apply_router`` call site — same FeatureBundle input, same ScoreResult
output, ~0.1ms inference.

Usage:
    uv run python scripts/train_p3.py --dates 2026-07-13,2026-07-14
    uv run python scripts/train_p3.py --dir logs --output models/p3_lightgbm.txt
    uv run python scripts/train_p3.py --auto          # auto-discover all dates
    uv run python scripts/train_p3.py --auto --holdout-days 7 --active-model models/p3_lightgbm.txt
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

# ── Cost-sensitive learning ──────────────────────────────────────────────────
# Routing is a *decision* problem, not a plain classification problem: the cost
# of a mistake is asymmetric. Under-routing (predicting a cheaper tier than the
# task needs) hurts answer quality; over-routing (predicting a pricier tier)
# only wastes money. A plain accuracy objective treats these equally and — worse
# — lets the majority classes (c0/c1) dominate, ignoring the rare-but-costly c3.
#
# We encode this as a cost matrix C[true, pred] and (a) weight training samples
# by how expensive their class is to get wrong, and (b) report `expected_cost`
# (mean per-sample misrouting cost) so the self-learn gate deploys the model
# that lowers *business cost*, not the one with the prettiest accuracy.
#
# Relative per-tier price (rough order-of-magnitude; override via ROUTER_TIER_PRICE
# JSON env or edit here). Used by --cost-mode tier-price.
DEFAULT_TIER_PRICE = {"c0": 1.0, "c1": 2.0, "c2": 4.0, "c3": 8.0}


def _tier_price() -> dict:
    """Per-tier relative price, overridable via ROUTER_TIER_PRICE JSON env."""
    raw = os.environ.get("ROUTER_TIER_PRICE", "").strip()
    if raw:
        try:
            data = json.loads(raw)
            return {t: float(data.get(t, DEFAULT_TIER_PRICE[t])) for t in TIERS}
        except (ValueError, TypeError):
            logger.warning("invalid ROUTER_TIER_PRICE JSON; using defaults")
    return dict(DEFAULT_TIER_PRICE)


def build_cost_matrix(
    cost_mode: str = "linear",
    under_mult: float = 3.0,
    over_mult: float = 1.0,
) -> np.ndarray:
    """Return C[true_idx, pred_idx] = cost of routing a `true` turn to `pred`.

    - ``linear``: cost = |pred - true|, scaled by under_mult when under-routing
      (pred < true) and over_mult when over-routing (pred > true). Correct = 0.
    - ``tier-price``: like linear but each step's cost is weighted by the price
      gap between the tiers, so jumping c0→c3 costs far more than c2→c3.

    under_mult > over_mult encodes "under-routing is worse than over-routing".
    """
    C = np.zeros((4, 4), dtype=np.float64)
    price = _tier_price()
    price_arr = np.array([price[t] for t in TIERS], dtype=np.float64)
    for t in range(4):
        for p in range(4):
            if p == t:
                continue
            dist = abs(p - t)
            if cost_mode == "tier-price":
                # Normalize the price gap by the cheapest tier so c0→c1 ≈ 1 step.
                dist = abs(price_arr[p] - price_arr[t]) / max(price_arr[0], 1e-9)
            C[t, p] = dist * (under_mult if p < t else over_mult)
    return C


def expected_cost(y_true: np.ndarray, y_pred: np.ndarray, C: np.ndarray) -> float:
    """Mean per-sample misrouting cost. 0.0 = perfect routing; higher = worse."""
    n = len(y_true)
    if n == 0:
        return 0.0
    return float(np.mean(C[y_true, y_pred]))


def class_weights(y: np.ndarray) -> np.ndarray:
    """Per-class weight ∝ 1/sqrt(count), normalized so mean weight = 1.

    Inverse-sqrt (not inverse-frequency) gives a *gentle* rebalance: it lifts
    the rare c3 (~2.6% of data) without letting a handful of samples dominate
    the loss. Returns an array of shape (4,) indexed by class.
    """
    counts = np.bincount(y, minlength=4).astype(np.float64)
    inv = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1.0)), 0.0)
    # Normalize so the average *present* class weight is 1.
    present = inv[inv > 0]
    if len(present):
        inv = inv / present.mean()
    return inv


def sample_weights(
    y: np.ndarray, C: np.ndarray, use_class_weight: bool = True
) -> np.ndarray:
    """Per-sample training weight = (class weight) × (cost of misrouting the class).

    The cost term is the max off-diagonal cost of that true class — how expensive
    it is to get the class wrong. Multiplying focuses capacity on the samples
    whose mistakes are most costly (rare, high-tier, easily under-routed).
    """
    w = np.ones(len(y), dtype=np.float64)
    if use_class_weight:
        cw = class_weights(y)
        w *= cw[y]
    # Per-class "cost of getting it wrong" = max off-diagonal cost for that row.
    row_cost = np.array([
        max((C[t, p] for p in range(4) if p != t), default=0.0) for t in range(4)
    ])
    if row_cost.max() > 0:
        w *= (1.0 + row_cost[y])
    # Normalize to mean 1 so learning_rate semantics stay familiar.
    if w.mean() > 0:
        w /= w.mean()
    return w



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


def _split_time_holdout(
    records: list[dict], holdout_days: int
) -> tuple[list[dict], list[dict]]:
    """Split records into (train_pool, holdout) by a trailing time window.

    Records carry ``ts_ms`` (decision time). Sorting by it and taking the most
    recent ``holdout_days`` of data as the holdout set makes the eval mimic the
    real deployment question: "does a model trained on the past generalize to
    the *future*?" — unlike a random split, which leaks distribution and can
    hide regressions. Records without a usable ts_ms go to the train pool.
    """
    if holdout_days <= 0:
        return records, []
    timed = [(int(r.get("ts_ms") or 0), r) for r in records]
    timed.sort(key=lambda t: t[0])
    ts_vals = [t for t, _ in timed if t > 0]
    if not ts_vals:
        return records, []
    latest = ts_vals[-1]
    cutoff = latest - holdout_days * 86400 * 1000
    train_pool = [r for t, r in timed if t <= cutoff or t == 0]
    holdout = [r for t, r in timed if t > cutoff]
    # Guard: a holdout with no usable labels or only one class is useless.
    if not holdout:
        return records, []
    return train_pool, holdout


def _print_eval(
    y_true: np.ndarray, y_pred: np.ndarray, label: str, C: np.ndarray | None = None
) -> None:
    """Print a compact classification report (confusion matrix + per-class stats).

    When a cost matrix ``C`` is given, also prints the per-class and overall
    expected misrouting cost — the business-aligned metric the gate optimizes.
    """
    n = len(y_true)
    if n == 0:
        print(f"  {label}: (no data)")
        return
    acc = float(np.mean(y_true == y_pred))
    if C is not None:
        cost = expected_cost(y_true, y_pred, C)
        print(f"  {label}: accuracy={acc:.1%}  expected_cost={cost:.3f}  n={n}")
    else:
        print(f"  {label}: accuracy={acc:.1%}  n={n}")

    # Confusion matrix.
    print(f"    {'':>8}  " + "  ".join(f"pred_{t:>2}" for t in TIERS))
    for i, tier in enumerate(TIERS):
        row = [int(np.sum((y_true == i) & (y_pred == j))) for j in range(len(TIERS))]
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

    if C is not None:
        # Where the cost is concentrated: which true classes bleed the most.
        print("    per-class expected cost (avg cost of misrouting this class):")
        for i, tier in enumerate(TIERS):
            mask = y_true == i
            if int(np.sum(mask)) == 0:
                continue
            c = expected_cost(y_true[mask], y_pred[mask], C)
            print(f"      {tier}: {c:.3f}")


def train(
    dates: list[str],
    capture_dir: str,
    output: str,
    auto: bool,
    val_ratio: float,
    num_boost_round: int,
    holdout_days: int = 7,
    active_model_path: str = "",
    min_holdout_gain: float = 0.0,
    cost_mode: str = "linear",
    under_mult: float = 3.0,
    over_mult: float = 1.0,
    use_class_weight: bool = True,
    gate_metric: str = "cost",
) -> dict:
    """Train a LightGBM 4-class tier classifier. Returns summary dict.

    Cost-sensitive: samples are weighted by (class weight) × (misrouting cost)
    and the model is evaluated on ``expected_cost`` (mean per-sample misrouting
    cost from the cost matrix), not just accuracy. Under-routing costs more
    than over-routing (``under_mult`` > ``over_mult``), so the learned boundary
    is pushed toward *not* under-serving hard turns — which is the business
    objective (quality first, cost second).

    When ``holdout_days > 0`` the most recent ``holdout_days`` of labeled data
    are held out by time (not randomly) and used to compare the new model
    against the currently-serving ``active_model_path``. ``gate_metric`` picks
    the comparison: ``"cost"`` (default — new model must LOWER expected_cost by
    ≥ ``min_holdout_gain``) or ``"accuracy"`` (new model must RAISE accuracy by
    ≥ ``min_holdout_gain``). ``promote_ok`` is True only when the gate passes.
    """
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

    # Cost matrix for weighting + evaluation.
    C = build_cost_matrix(cost_mode=cost_mode, under_mult=under_mult, over_mult=over_mult)

    # Time-window holdout: most recent N days are the true eval set.
    train_pool, holdout_records = _split_time_holdout(all_records, holdout_days)
    X_hold, y_hold = _extract_xy(holdout_records)

    X, y = _extract_xy(train_pool)
    n = len(y)
    tier_counts = Counter(y.tolist())

    print(f"\n{'='*60}")
    print(f"  P3 LightGBM Training (cost-sensitive)")
    print(f"{'='*60}")
    print(f"  Dates:            {', '.join(dates)}")
    print(f"  Labeled records:  {n} (train pool) + {len(y_hold)} (holdout last {holdout_days}d)")
    print(f"  Tier distribution: {dict((TIERS[k], v) for k, v in sorted(tier_counts.items()))}")
    print(f"  Cost mode:        {cost_mode} (under×{under_mult} / over×{over_mult}, "
          f"class_weight={'on' if use_class_weight else 'off'})")

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

    # Cost-sensitive per-sample weights for the train split.
    w_train = sample_weights(y_train, C, use_class_weight=use_class_weight)
    cw = class_weights(y_train)
    print(f"  Class weights:    {dict((TIERS[i], round(float(cw[i]), 3)) for i in range(4))}")

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

    train_set = lgb.Dataset(
        X_train, label=y_train, weight=w_train, feature_name=list(FEATURE_ORDER)
    )
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
    _print_eval(y_train, y_pred_train, "TRAIN", C)
    print()
    _print_eval(y_val, y_pred_val, "VAL  ", C)

    # ── Time-window holdout eval + comparison vs the active model ──
    holdout_acc: float | None = None
    active_acc: float | None = None
    holdout_cost: float | None = None
    active_cost: float | None = None
    delta: float | None = None
    cost_delta: float | None = None
    promote_ok = False
    if len(y_hold) > 0:
        y_pred_hold = np.argmax(model.predict(X_hold), axis=1)
        holdout_acc = float(np.mean(y_hold == y_pred_hold))
        holdout_cost = expected_cost(y_hold, y_pred_hold, C)
        print()
        _print_eval(y_hold, y_pred_hold, f"HOLDOUT (last {holdout_days}d)", C)
        # Compare against the currently-serving model on the SAME future window.
        if active_model_path and os.path.exists(active_model_path):
            try:
                active_model = lgb.Booster(model_file=active_model_path)
                y_pred_active = np.argmax(active_model.predict(X_hold), axis=1)
                active_acc = float(np.mean(y_hold == y_pred_active))
                active_cost = expected_cost(y_hold, y_pred_active, C)
                delta = round(holdout_acc - active_acc, 6)
                cost_delta = round(active_cost - holdout_cost, 6)  # +ve = new model cheaper
                print(f"\n  Active model:   acc={active_acc:.1%}  expected_cost={active_cost:.3f}")
                print(f"  New model:      acc={holdout_acc:.1%}  expected_cost={holdout_cost:.3f}")
                if gate_metric == "cost":
                    promote_ok = cost_delta >= min_holdout_gain
                    print(f"  Cost delta (active-new): {cost_delta:+.3f}  (gate >= {min_holdout_gain:+.3f})")
                else:
                    promote_ok = delta >= min_holdout_gain
                    print(f"  Acc delta (new-active):  {delta:+.2%}  (gate >= {min_holdout_gain:+.2%})")
                print(f"  Gate[{gate_metric}]: {'PASS — promote' if promote_ok else 'FAIL — keep active'}")
            except Exception as exc:
                logger.warning("active model eval failed (%s); skipping comparison", exc)
        else:
            # No active model to beat → promoting is safe (first deploy).
            promote_ok = True
            if active_model_path:
                print(f"\n  Active model not found at {active_model_path}; treating as first deploy.")

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
        "holdout_days": holdout_days,
        "holdout_samples": len(y_hold),
        "holdout_accuracy": holdout_acc,
        "active_model_path": active_model_path or None,
        "active_holdout_accuracy": active_acc,
        "holdout_delta": delta,
        "cost_mode": cost_mode,
        "under_mult": under_mult,
        "over_mult": over_mult,
        "class_weight": use_class_weight,
        "gate_metric": gate_metric,
        "holdout_cost": holdout_cost,
        "active_holdout_cost": active_cost,
        "cost_delta": cost_delta,
        "promote_ok": promote_ok,
        "schema_version": 1,
        "trained_on": dates,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Model saved: {output}")
    print(f"  Metadata:    {meta_path}")
    print(f"{'='*60}")

    return {
        "trained": True,
        "n_samples": n,
        "val_accuracy": float(np.mean(y_val == y_pred_val)),
        "holdout_accuracy": holdout_acc,
        "active_holdout_accuracy": active_acc,
        "holdout_delta": delta,
        "holdout_cost": holdout_cost,
        "active_holdout_cost": active_cost,
        "cost_delta": cost_delta,
        "promote_ok": promote_ok,
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
    p.add_argument("--holdout-days", type=int, default=7,
                   help="hold out the most recent N days as the future eval window (0=disable)")
    p.add_argument("--active-model", default="",
                   help="path to the currently-serving model to beat on the holdout window")
    p.add_argument("--min-holdout-gain", type=float, default=0.0,
                   help="min holdout gain over active model required to pass the gate "
                        "(expected_cost reduction for --gate-metric cost, accuracy gain for accuracy)")
    p.add_argument("--gate-metric", choices=["cost", "accuracy"], default="cost",
                   help="metric the promote gate optimizes (default: cost)")
    p.add_argument("--cost-mode", choices=["linear", "tier-price"], default="linear",
                   help="cost matrix mode: tier-distance, or tier-price-gap weighted")
    p.add_argument("--under-mult", type=float, default=3.0,
                   help="cost multiplier for under-routing (default 3.0 — quality first)")
    p.add_argument("--over-mult", type=float, default=1.0,
                   help="cost multiplier for over-routing (default 1.0)")
    p.add_argument("--no-class-weight", dest="class_weight", action="store_false",
                   help="disable the inverse-sqrt class rebalance")
    p.set_defaults(class_weight=True)
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
        holdout_days=args.holdout_days,
        active_model_path=args.active_model,
        min_holdout_gain=args.min_holdout_gain,
        cost_mode=args.cost_mode,
        under_mult=args.under_mult,
        over_mult=args.over_mult,
        use_class_weight=args.class_weight,
        gate_metric=args.gate_metric,
    )
    return 0 if summary.get("trained") else 1


if __name__ == "__main__":
    raise SystemExit(main())
