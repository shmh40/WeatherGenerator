#!/usr/bin/env python3
"""Check val results for one or more run IDs."""
import json, sys, os

results_dir = "/hpcperm/ecm8347/work/wg_autoresearch/WeatherGenerator/results"
output_dir = "/hpcperm/ecm8347/work/wg_autoresearch/WeatherGenerator/output"

run_ids = sys.argv[1:]
if not run_ids:
    # Auto-discover from results directory
    if os.path.isdir(results_dir):
        run_ids = sorted(d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)))

for run_id in run_ids:
    print(f"=== {run_id} ===")
    f = os.path.join(results_dir, run_id, f"{run_id}_train_metrics.json")
    if not os.path.isfile(f):
        print("  no results yet")
        continue
    with open(f) as fh:
        lines = fh.readlines()
    vals = [json.loads(l) for l in lines if json.loads(l).get("stage") == "val"]
    trains = [json.loads(l) for l in lines if json.loads(l).get("stage") == "train"]
    print(f"  vals={len(vals)} trains={len(trains)}")
    for i, v in enumerate(vals):
        loss = v.get("LossPhysical.loss_avg", v.get("loss.LossPhysical.ERA5.mse.loss_avg", "?"))
        marker = " <-- LATEST" if i == len(vals) - 1 else ""
        print(f"  VAL {i+1}: loss={loss:.6f}{marker}" if isinstance(loss, float) else f"  VAL {i+1}: loss={loss}{marker}")

    # Find peak_vram_mb from any line
    peak_vram = None
    for l in lines:
        d = json.loads(l)
        v = d.get("peak_vram_mb")
        if v and v > 0:
            peak_vram = v
    if peak_vram:
        print(f"  peak VRAM: {peak_vram:.0f} MB ({peak_vram/1024:.1f} GB)")

    if trains:
        d = trains[-1]
        ns = d.get("num_samples", "?")
        loss = d.get("LossPhysical.loss_avg", d.get("loss_avg_mean", "?"))
        lr = d.get("learning_rate", "?")
        print(f"  latest train: samples={ns} loss={loss:.6f} lr={lr:.2e}" if isinstance(loss, float) and isinstance(lr, float) else f"  latest train: samples={ns}")
    print()
