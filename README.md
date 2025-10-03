
# C51 Project — A Clean, Reproducible Distributional DQN (C51) Baseline

Minimal, readable reproduction of **C51** vs a matched **DQN** baseline, with optional **n-step** returns and ablations (atoms, support bounds, eval protocol). Focus: **determinism, simple configs, and plot-ready outputs**.

---

## TL;DR (CartPole-v1, 200k steps; eval=15 episodes; 3 seeds)

- **C51 beats DQN on sample efficiency** and stability; **C51 (n=3)** is best.
- Headline metric **steps_to_195** (first training step where eval average ≥ 195):
  - **DQN:** 40170.7 ± 2.08
  - **C51 (n=1):** 11875.0 ± 448.6
  - **C51 (n=3):** **7987.0 ± 1250.5**
- Practical defaults from ablations: **support ≈ [0,120]**, **~51 atoms**, **n=3**, **15 eval episodes**.

> Full report text folded into this README; figures can be regenerated from the logged CSVs/W&B runs.

---

## Quickstart

### 1) Environment
```bash
# Create/activate a venv (rename as you like)
python3 -m venv c51_venv
source c51_venv/bin/activate  # Windows: c51_venv\Scripts\activate

# Upgrade pip and install
pip install --upgrade pip
pip install -r requirements.txt
```

**Python:** 3.10–3.12  
**Hardware:** CPU is fine for CartPole; GPU optional.

### 2) Train
```bash
# C51 on CartPole-v1 (default config; n=1)
python -m src.train --algo c51 --env CartPole-v1 --total-steps 200000 --eval-every 10000 --eval-episodes 15 --seed 0

# DQN baseline
python -m src.train --algo dqn --env CartPole-v1 --total-steps 200000 --eval-every 10000 --eval-episodes 15 --seed 0

# C51 with n-step returns (e.g., n=3)
python -m src.train --algo c51 --n-step 3 --env CartPole-v1 --total-steps 200000 --eval-every 10000 --eval-episodes 15 --seed 0
```

Common flags:
```
--algo {dqn,c51}      --env <Gymnasium id>      --seed <int>
--total-steps <int>   --eval-episodes <int>     --eval-every <steps>
--atoms <int>         --vmin <float>            --vmax <float>
--n-step <int>        --buffer-size <int>       --batch-size <int>
--lr <float>          --gamma <float>           --tau <float>  # target update
--save-dir results/YYYY-MM-DD/
```

### 3) Evaluate / Plot
```bash
# Evaluate a saved checkpoint (averaged score; steps_to_threshold if applicable)
python -m src.evaluate --run-dir results/2025-10-02/c51-CartPole-v1-seed0

# Make plots from one or more runs
python -m src.plots --runs results/2025-10-02/c51-* --out plots/2025-10-02
```

Outputs live under `results/<date>/<run_name>/`:
```
metrics.json            # scalar summary (means, CIs)
eval_history.csv        # per-interval evals
events.log              # config + seed record
learning_curve.png      # return vs steps
steps_to_threshold.csv  # (optional) first step achieving threshold
```

---

## Methods (what these scripts implement)

### Algorithms
- **DQN (baseline):** 2-layer MLP (128–128, ReLU), Huber loss, uniform replay; ε-greedy; target network with periodic copy.
- **C51:** same backbone; head outputs (A × Z) logits with log_softmax over atoms; expectation over fixed support yields Q(s, a). **Double-DQN** selection (argmax w/ online net; distribution from target net); cross-entropy loss to the **projected distributional** target.
- **n-step:** experience relabeled with n-step returns (targets use γⁿ).

### Environment & Budgets
- **Task:** CartPole-v1 (reward +1/step until termination).  
- **Training budget:** 200k environment steps per run (headline).  
- **Evaluation:** greedy policy for **15 episodes** per eval event (sensitivity with 5/25 also reported).  
- **Seeds:** 3 independent seeds for all headline configs/ablations (except sensitivity runs where noted).  
- **C51 support:** ablated over **[0,50]**, **[0,120] (headline)**, **[0,150]**.  
- **Atoms:** ablated over **{21, 51, 101}** (headline **51**).

### Metric
- **steps_to_195 (CartPole):** first global training step where the logged **eval average** ≥ 195.

---

## Results — CartPole (200k steps; eval=15; 3 seeds)

### Headline comparison
Per-run **steps_to_195** and summary (mean ± std).

| Method       | Per-seed steps_to_195      | Mean ± Std |
|---|---|---|
| DQN          | 40169, 40173, 40170        | **40170.7 ± 2.08** |
| C51 (n=1)    | 11616, 11616, 12393        | **11875.0 ± 448.6** |
| C51 (n=3)    | 8709, 8709, 6543           | **7987.0 ± 1250.5** |

**Takeaway:** C51 improves sample efficiency vs DQN; **C51 (n=3)** improves further and yields smoother eval curves.

### n-step sweep (C51, n ∈ {1,3,5})
| n | Per-seed steps_to_195 | Mean ± Std |
|---:|---|---|
| 1 | 11616, 11616, 12393 | **11875.0 ± 448.6** |
| 3 | 8709, 8709, 6543    | **7987.0 ± 1250.5** |
| 5 | 14897, 15958, 16965 | **15940.0 ± 1034.1** |

**Observation:** **n=3** balances bias/variance and reaches threshold earlier than n=1 or n=5 under the same budget.

### Support sweep (C51, n=3; support ∈ {[0,50], [0,120], [0,150]})
| Support | Per-seed steps_to_195 | Mean ± Std |
|---|---|---|
| [0, 50]  | 26705, 26930, 26710 | **26781.7 ± 128.5** |
| [0, 120] | 8709, 8709, 6543    | **7987.0 ± 1250.5** |
| [0, 150] | 46500, 46505, 46508 | **46504.3 ± 4.04** |

**Analysis:** Aligning the support to the return scale (≈[0,120]) avoids projection clamping (too narrow) and preserves useful resolution (too wide hurts early learning).

### Atom count (C51, n=3; atoms ∈ {21, 51, 101})
| Atoms | Per-seed steps_to_195 | Mean ± Std |
|---:|---|---|
| 21  | 8756, 8757, 8757       | **8756.7 ± 0.58** |
| 51  | 8709, 8709, 6543       | **7987.0 ± 1250.5** |
| 101 | 29150, 28793, 29496    | **29146.3 ± 351.5** |

**Observation:** **~51 atoms** is a strong default on CartPole; 21 is close, while 101 increases compute/memory with no gain here.

### Evaluation-episode sensitivity (single-seed)
| Method | 5 episodes | 15 episodes (baseline) | 25 episodes |
|---|---:|---:|---:|
| C51 (n fixed) | 8709 | 8709 (median of the 3-seed set) | 8600 |
| DQN           | 40170 | 40169–40173 (3 seeds) | 40169 |

**Conclusion:** 15–25 episodes slightly reduce variance without changing the ordering **DQN < C51 < C51+n=3**.

---

## Why it works (intuitions)
- **Why C51 helps:** distributional targets provide richer gradients than scalar TD and better capture multi-modal returns → improved stability and sample efficiency.  
- **Why n=3 helps:** densifies credit in early learning (bias/variance trade-off beats n=1 and n=5 at fixed budget).  
- **Why support matters:** match to the return scale to avoid clamping (too narrow) or resolution loss (too wide).  
- **Why ~51 atoms:** practical sweet spot for resolution vs compute on small control.

---

## Reproducibility checklist
- **Code & configs:** https://github.com/quguanni/c51_project  
- **W&B project:** https://wandb.ai/quguanni-california-institute-of-technology-caltech/c51-project?nw=nwuserquguanni  
- **Envs:** CartPole-v1 (200k steps). LunarLander-v3 code is included; runs are ongoing.  
- **Metric:** `steps_to_195` (eval avg ≥ 195).  
- **Hardware:** MacBook Pro M3 Max (16-core CPU, 40-core integrated GPU, 48 GB unified memory).  
- **OS/Kernel:** macOS Sequoia 15.6; Darwin 24.6.0.  
- **Python/W&B:** 3.12.4 / 0.19.9.  
- **Figures:** exported from W&B compare view with fixed y-axis & smoothing; plots also generated by `src.plots` from CSVs.

---

## Project Structure
```
c51_project/
├─ src/
│  ├─ agents/
│  │  ├─ dqn.py
│  │  ├─ c51.py
│  │  └─ nets.py
│  ├─ buffers/replay_buffer.py
│  ├─ envs/make_env.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ plots.py
│  └─ utils/{seed.py,log.py,metrics.py,checkpoint.py}
├─ configs/ (optional yaml configs)
├─ tests/ (unit tests; golden expected outputs)
├─ results/
├─ plots/
└─ requirements.txt
```

---

## Tests
```bash
pytest -q
```
- Golden test: learn a toy environment for a few thousand steps and verify return > threshold.  
- Static tests: shapes, distributional projection, atom support.

---

## Appendix A — Hyperparameters (CartPole headline)
- **DQN:** lr=3e-4, batch=128, buffer=120k, γ=0.99, ε: 1.0→0.01 over 25k steps, target copy every 500 steps.  
- **C51 (n=1):** n_atoms=51; support [0,120]; cross-entropy loss; Double-DQN selection.  
- **C51 (n=3):** as above with n_step=3 and γ³ in the distributional target.  
- **steps_to_195:** first step such that the **eval** episodic average return ≥ 195.

> Exact values (and the ablation grids) live in the repo configs.

---

## License
MIT (see `LICENSE`).


