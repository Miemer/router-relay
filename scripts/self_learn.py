"""P3 self-learning loop: one idempotent end-to-end cycle.

Chains the existing offline pieces into a single automated run:

    realign_labels  →  judge_labels (optional)  →  train_p3 (time-holdout + gate)
                    →  promote via registry + hot reload (if the gate passes)

It is designed to be triggered by an external scheduler (cron / Windows Task
Scheduler / a container orchestrator) once per day off-peak, e.g.::

    uv run python scripts/self_learn.py --auto --holdout-days 7

Every step is idempotent and safe to re-run: realign rewrites the labeled file
for a date, judge skips already-judged decisions (``--skip-judged``), training
is deterministic (fixed seed), and promotion only happens when the new model
beats the currently-serving model on the future holdout window by the required
margin. A failure at any step leaves the previously-serving model untouched.

The registry (``models/registry.json``) is the deploy mechanism: promoting
flips the active pointer to the new version, and the running server picks it up
on the next routing decision (hot reload — no restart). Older versions are kept
so a one-line pointer flip rolls back.

Flags:
    --date YYYY-MM-DD     realign/judge a single date (default: today)
    --auto                realign/judge/train across all captured dates
    --no-judge            skip the LLM-as-judge step (faster, weaker labels)
    --max-judge N         cap judge calls this run (cost control)
    --holdout-days N      future eval window for the gate (default 7)
    --min-holdout-gain F  required holdout gain over the active model (default 0.0)
    --dry-run             print the plan without calling the judge or promoting
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime

# Make the relay package + sibling scripts importable (standalone script).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "src"))
sys.path.insert(0, _SCRIPT_DIR)

from relay.router.registry import (  # noqa: E402
    ModelRegistry,
    ModelVersion,
    load_registry,
    new_version_id,
    registry_path_for,
    versions_dir_for,
)
import realign_labels  # noqa: E402
import train_p3  # noqa: E402

logger = logging.getLogger("self_learn")


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _discover_sample_dates(capture_dir: str) -> list[str]:
    """All dates that have a router-samples file (i.e. captured traffic)."""
    pattern = os.path.join(capture_dir, "router-samples-*.jsonl")
    return sorted(
        os.path.basename(p).replace("router-samples-", "").replace(".jsonl", "")
        for p in glob.glob(pattern)
    )


def _run_realign(dates: list[str], capture_dir: str) -> dict:
    """Realign labels for each date; returns per-date summaries."""
    summaries = {}
    for date in dates:
        try:
            summaries[date] = realign_labels.realign(date, capture_dir, dry_run=False)
        except Exception as exc:
            logger.warning("realign failed for %s (%s); skipped", date, exc)
    return summaries


def _run_judge(dates: list[str], capture_dir: str, model: str, max_judge: int | None,
               dry_run: bool) -> dict:
    """Run the LLM-as-judge on un-judged decisions for each date (cost-capped).

    Imported lazily so a missing upstream config doesn't break judge-free runs.
    """
    import judge_labels  # local import: requires upstream client config

    totals = {"judged": 0, "failed": 0}
    for date in dates:
        try:
            summary = judge_labels.asyncio.run(judge_labels.run_judge(
                date=date,
                capture_dir=capture_dir,
                model=model,
                limit=max_judge,
                delay=0.5,
                max_chars=8000,
                dry_run=dry_run,
                skip_judged=True,
                judge_timeout=60.0,
            ))
            totals["judged"] += summary.get("judged", 0)
            totals["failed"] += summary.get("failed", 0)
        except Exception as exc:
            logger.warning("judge failed for %s (%s); continuing without it", date, exc)
    return totals


def _promote(models_dir: str, train_summary: dict, dates: list[str]) -> str | None:
    """Copy the trained model into versions/ and flip the registry active pointer.

    Returns the new version id, or None if promotion was skipped.
    """
    output = train_summary.get("output")
    if not output or not os.path.exists(output):
        return None

    version_id = new_version_id()
    vdir = versions_dir_for(models_dir)
    os.makedirs(vdir, exist_ok=True)
    model_dst = os.path.join(vdir, f"{version_id}.txt")
    meta_dst = os.path.join(vdir, f"{version_id}.meta.json")
    shutil.copy2(output, model_dst)
    src_meta = output.replace(".txt", ".meta.json")
    if not src_meta.endswith(".json"):
        src_meta = output + ".meta.json"
    if os.path.exists(src_meta):
        shutil.copy2(src_meta, meta_dst)

    reg: ModelRegistry = load_registry(models_dir)
    reg.add(ModelVersion(
        version=version_id,
        model_path=model_dst,
        meta_path=meta_dst,
        trained_on=dates,
        num_samples=int(train_summary.get("n_samples") or 0),
        val_accuracy=float(train_summary.get("val_accuracy") or 0.0),
        holdout_accuracy=float(train_summary.get("holdout_accuracy") or 0.0),
        trained_ts_ms=int(time.time() * 1000),
    ), activate=True)
    return version_id


def _cleanup_candidate(candidate_output: str) -> None:
    """Best-effort removal of the temp candidate artifacts.

    The versioned copy in ``models/versions/`` is the source of truth, so the
    flat candidate files are redundant. Removal may be blocked by the host
    sandbox's safe-delete policy; that is non-fatal (the next run overwrites
    them), so failures are logged and ignored.
    """
    for p in (candidate_output, candidate_output.replace(".txt", ".meta.json")):
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError as exc:
            logger.info("could not remove candidate %s (%s); leaving in place", p, exc)



def run(args: argparse.Namespace) -> dict:
    capture_dir = args.capture_dir
    models_dir = args.models_dir
    os.makedirs(models_dir, exist_ok=True)

    # Resolve the set of dates to process.
    if args.auto:
        dates = _discover_sample_dates(capture_dir)
    else:
        dates = [args.date]
    if not dates:
        print("No captured dates found. Run the relay with ROUTER_CAPTURE_DIR set first.")
        return {"promoted": False, "reason": "no_data"}

    # The currently-serving model the new one must beat on the future window.
    reg = load_registry(models_dir)
    active = reg.active

    # Bootstrap: if the registry is empty but a serving model already exists
    # (legacy ROUTER_ML_MODEL_PATH default), register it as the active baseline
    # first. Without this, the very first self-learn run sees "no active model"
    # and promote_ok=True — silently replacing a stronger incumbent with an
    # unverified candidate. The baseline must be registered so the candidate
    # has to *beat* it on the holdout window.
    baseline_model = args.active_model or "models/p3_lightgbm.txt"
    if active is None and os.path.exists(baseline_model):
        baseline_id = new_version_id()
        vdir = versions_dir_for(models_dir)
        os.makedirs(vdir, exist_ok=True)
        model_dst = os.path.join(vdir, f"{baseline_id}.txt")
        meta_dst = os.path.join(vdir, f"{baseline_id}.meta.json")
        shutil.copy2(baseline_model, model_dst)
        src_meta = baseline_model.replace(".txt", ".meta.json")
        if os.path.exists(src_meta):
            shutil.copy2(src_meta, meta_dst)
        reg.add(ModelVersion(
            version=baseline_id,
            model_path=model_dst,
            meta_path=meta_dst,
            trained_on=[],
            trained_ts_ms=int(time.time() * 1000),
        ), activate=True)
        active = reg.active
        print(f"  Bootstrapped baseline {baseline_id} from {baseline_model}")

    active_path = (active.model_path if active else "") or ""

    print(f"\n{'='*64}")
    print(f"  P3 Self-Learning Cycle")
    print(f"{'='*64}")
    print(f"  Dates:            {', '.join(dates)}")
    print(f"  Capture dir:      {capture_dir}")
    print(f"  Models dir:       {models_dir}")
    print(f"  Active model:     {active.version if active else '(none — first deploy)'}")
    print(f"  Judge:            {'disabled' if args.no_judge else f'enabled (cap {args.max_judge or "∞"})'}")
    print(f"  Holdout window:   last {args.holdout_days}d  (gate >= {args.min_holdout_gain:+.2%})")
    if args.dry_run:
        print(f"  Mode:             DRY RUN")
    print(f"{'='*64}\n")

    # 1. Realign labels (join decisions + outcomes + judge → labeled).
    print("[1/4] Realigning labels...")
    realign_summaries = _run_realign(dates, capture_dir)
    labeled_total = sum(s.get("labeled", 0) for s in realign_summaries.values())
    print(f"      labeled so far: {labeled_total}")

    # 2. LLM-as-judge for un-judged decisions (optional, cost-capped).
    judge_totals = {"judged": 0, "failed": 0}
    if not args.no_judge:
        print("[2/4] Judging un-labeled turns (LLM-as-judge)...")
        judge_totals = _run_judge(dates, capture_dir, args.judge_model,
                                  args.max_judge, args.dry_run)
        print(f"      judged={judge_totals['judged']} failed={judge_totals['failed']}")
        # Re-realign so fresh judge labels are folded into the labeled files.
        if judge_totals["judged"] and not args.dry_run:
            print("      re-realigning to fold in judge labels...")
            realign_summaries = _run_realign(dates, capture_dir)
            labeled_total = sum(s.get("labeled", 0) for s in realign_summaries.values())
            print(f"      labeled now: {labeled_total}")
    else:
        print("[2/4] Judge skipped (--no-judge).")

    if args.dry_run:
        print("\n[dry-run] would train on the labeled data and evaluate the gate; "
              "no model trained, nothing promoted.")
        return {"promoted": False, "reason": "dry_run", "labeled": labeled_total}

    # 3. Train with a time-window holdout + compare against the active model.
    print("[3/4] Training (time-window holdout + gate)...")
    candidate_output = os.path.join(models_dir, "_candidate.txt")
    train_summary = train_p3.train(
        dates=dates,
        capture_dir=capture_dir,
        output=candidate_output,
        auto=False,
        val_ratio=args.val_ratio,
        num_boost_round=args.num_boost_round,
        holdout_days=args.holdout_days,
        active_model_path=active_path,
        min_holdout_gain=args.min_holdout_gain,
        cost_mode=args.cost_mode,
        under_mult=args.under_mult,
        over_mult=args.over_mult,
        use_class_weight=args.class_weight,
        gate_metric=args.gate_metric,
    )
    if not train_summary.get("trained"):
        print("      training aborted (insufficient data). Keeping active model.")
        return {"promoted": False, "reason": "insufficient_data",
                "labeled": labeled_total}

    # 4. Gate → promote via registry (hot reload) or keep active.
    print("[4/4] Evaluating gate...")
    gate_metric = args.gate_metric
    cost_delta = train_summary.get("cost_delta")
    acc_delta = train_summary.get("holdout_delta")
    # Human-readable delta matching the gate metric actually optimized.
    if gate_metric == "cost":
        delta_txt = (f"cost Δ {cost_delta:+.3f}" if cost_delta is not None
                     else "n/a (first deploy)")
    else:
        delta_txt = (f"acc Δ {acc_delta:+.2%}" if acc_delta is not None
                     else "n/a (first deploy)")
    if train_summary.get("promote_ok"):
        version_id = _promote(models_dir, train_summary, dates)
        print(f"      GATE PASS[{gate_metric}] → promoted {version_id} ({delta_txt})")
        print(f"      server picks it up on the next routing decision (hot reload).")
        # Clean the candidate artifact (the versioned copy is the source of truth).
        _cleanup_candidate(candidate_output)
        return {
            "promoted": True,
            "version": version_id,
            "gate_metric": gate_metric,
            "holdout_accuracy": train_summary.get("holdout_accuracy"),
            "active_holdout_accuracy": train_summary.get("active_holdout_accuracy"),
            "holdout_delta": acc_delta,
            "holdout_cost": train_summary.get("holdout_cost"),
            "active_holdout_cost": train_summary.get("active_holdout_cost"),
            "cost_delta": cost_delta,
            "labeled": labeled_total,
            "judged": judge_totals["judged"],
        }
    else:
        print(f"      GATE FAIL[{gate_metric}] → keeping active model ({delta_txt}, "
              f"needed >= {args.min_holdout_gain:+.3f})")
        _cleanup_candidate(candidate_output)
        return {
            "promoted": False,
            "reason": "gate_failed",
            "gate_metric": gate_metric,
            "holdout_accuracy": train_summary.get("holdout_accuracy"),
            "active_holdout_accuracy": train_summary.get("active_holdout_accuracy"),
            "holdout_delta": acc_delta,
            "holdout_cost": train_summary.get("holdout_cost"),
            "active_holdout_cost": train_summary.get("active_holdout_cost"),
            "cost_delta": cost_delta,
            "labeled": labeled_total,
            "judged": judge_totals["judged"],
        }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P3 self-learning loop (realign → judge → train → gate → promote).")
    p.add_argument("--date", default=_today(), help="YYYY-MM-DD (default: today); ignored with --auto")
    p.add_argument("--auto", action="store_true", help="process all captured dates")
    p.add_argument("--dir", dest="capture_dir", default="logs", help="capture directory")
    p.add_argument("--models-dir", default="models", help="models dir (registry + versions)")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM-as-judge step")
    p.add_argument("--judge-model", default="gpt-5.6-terra", help="judge model id")
    p.add_argument("--max-judge", type=int, default=None, help="cap judge calls this run")
    p.add_argument("--active-model", default="", help="override active model path (else registry)")
    p.add_argument("--holdout-days", type=int, default=7, help="future eval window for the gate")
    p.add_argument("--min-holdout-gain", type=float, default=0.0,
                   help="required holdout gain over active model to promote")
    p.add_argument("--val-ratio", type=float, default=0.2, help="train/val split ratio")
    p.add_argument("--num-boost-round", type=int, default=200, help="boosting rounds")
    p.add_argument("--gate-metric", choices=["cost", "accuracy"], default="cost",
                   help="metric the promote gate optimizes (default: cost)")
    p.add_argument("--cost-mode", choices=["linear", "tier-price"], default="linear",
                   help="cost matrix mode")
    p.add_argument("--under-mult", type=float, default=3.0,
                   help="cost multiplier for under-routing (quality first)")
    p.add_argument("--over-mult", type=float, default=1.0,
                   help="cost multiplier for over-routing")
    p.add_argument("--no-class-weight", dest="class_weight", action="store_false",
                   help="disable the inverse-sqrt class rebalance")
    p.set_defaults(class_weight=True)
    p.add_argument("--dry-run", action="store_true", help="plan only; no judge calls, no promote")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    summary = run(args)
    print(f"\n{'='*64}")
    print(f"  Self-learning cycle complete")
    print(f"{'='*64}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    # Exit 0 when a model is serving (promoted or kept); 1 only when nothing ran.
    return 0 if summary.get("promoted") or summary.get("reason") in (
        "gate_failed", "dry_run") else 1


if __name__ == "__main__":
    raise SystemExit(main())
