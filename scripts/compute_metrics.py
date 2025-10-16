# scripts/compute_metrics.py
import argparse, json, math, os, csv
import numpy as np

def load_eval_csv(path):
    with open(path) as f:
        r = csv.DictReader(f)
        rows = list(r)
        if not rows:
            return [], []
        # robust column detection
        step_key = next((k for k in rows[0].keys() if k.lower().endswith("step")), "step")
        eval_key = next((k for k in rows[0].keys() if "eval_return" in k.lower()), "eval_return")
        steps, vals = [], []
        for row in rows:
            try:
                s = float(row[step_key]); v = float(row[eval_key])
                if math.isnan(s) or math.isnan(v): continue
                steps.append(s); vals.append(v)
            except Exception:
                pass
        z = sorted(zip(steps, vals))
        steps, vals = zip(*z) if z else ([], [])
        return list(steps), list(vals)

def steps_to_threshold(steps, vals, thr):
    for s, v in zip(steps, vals):
        if v >= thr:
            return s
    return None

def auc_trapz(steps, vals, budget):
    if not steps: return 0.0
    A = 0.0
    for i in range(1, len(steps)):
        s0, s1 = steps[i-1], min(steps[i], budget)
        if s0 >= budget: break
        A += 0.5 * (vals[i-1] + vals[i]) * (s1 - s0)
        if steps[i] >= budget: break
    return A

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Folder containing eval_history.csv")
    ap.add_argument("--threshold", type=float, required=True, help="e.g., 195 for CartPole")
    ap.add_argument("--budget", type=float, required=True, help="e.g., 200000 for CartPole")
    ap.add_argument("--out", default="metrics.json")
    args = ap.parse_args()

    csv_path = os.path.join(args.run_dir, "eval_history.csv")
    steps, vals = load_eval_csv(csv_path)
    stt = steps_to_threshold(steps, vals, args.threshold)
    auc = auc_trapz(steps, vals, args.budget)
    best = float(max(vals)) if vals else None

    out = {
        "threshold": args.threshold,
        "budget": args.budget,
        "steps_to_threshold": stt,
        "AUC": auc,
        "best_eval": best,
    }
    with open(os.path.join(args.run_dir, args.out), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
