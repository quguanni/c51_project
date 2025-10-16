# scripts/aggregate_metrics.py
import argparse, json, os, numpy as np

def load_metric(run_dir):
    with open(os.path.join(run_dir, "metrics.json")) as f:
        m = json.load(f)
    return m["steps_to_threshold"], m["AUC"], m["best_eval"]

def fmt(mu, sd, digits=1):
    return f"{mu:.{digits}f} ± {sd:.{digits}f}"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="List of run dirs (same method, different seeds)")
    args = ap.parse_args()

    stt, auc, best = [], [], []
    for rd in args.runs:
        s, a, b = load_metric(rd)
        if s is not None: stt.append(s)
        if a is not None: auc.append(a)
        if b is not None: best.append(b)

    stt_mu, stt_sd = np.mean(stt), np.std(stt, ddof=1) if len(stt) > 1 else (np.nan)
    auc_mu, auc_sd = np.mean(auc), np.std(auc, ddof=1) if len(auc) > 1 else (np.nan)
    best_mu, best_sd = np.mean(best), np.std(best, ddof=1) if len(best) > 1 else (np.nan)

    out = {
        "steps_to_threshold": {"mean": stt_mu, "std": stt_sd},
        "AUC": {"mean": auc_mu, "std": auc_sd},
        "best_eval": {"mean": best_mu, "std": best_sd},
    }
    print(json.dumps(out, indent=2))
