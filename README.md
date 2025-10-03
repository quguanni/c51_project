# Distributional RL Repro (DQN → C51 / scaffold for QR-DQN)

A **clean, reproducible** baseline for small-task RL with:
- **DQN** baselines on CartPole & LunarLander
- **C51** distributional DQN on both tasks
- Seeds, CSV logs, simple plots, and ablation hooks (support bounds, atoms, n-step)

This repo favors **repro discipline** (fixed seeds, simple logs) over “hero runs.”

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Minimal single-seed runs (writes CSV to ./runs/<exp>/metrics.csv)
python dqn_cartpole.py
python dqn_lander.py
python c51_cartpole.py
python c51_lander.py

MIT License

Copyright (c) 2025 Guanni Qu

Permission is hereby granted, free of charge, to any person obtaining a copy...
[standard MIT text continues]

---
c51_project/
├── training.py
├── wandb_test.py
├── checkpoints/
│   ├── LunarLander-v3_c51_step{...}.pt
│   └── dqn_step{...}.pt
├── wandb/
│   ├── run-YYYYMMDD_HHMMSS-<id>/
│   │   ├── files/
│   │   │   ├── config.yaml
│   │   │   ├── requirements.txt
│   │   │   ├── wandb-metadata.json
│   │   │   ├── wandb-summary.json
│   │   │   ├── output.log
│   │   │   ├── checkpoints/  -> symlinks to ../checkpoints/*.pt
│   │   │   └── media/
│   │   │       └── images/
│   │   │           └── dist_snapshot_*.png
│   │   ├── logs/
│   │   │   ├── debug.log
│   │   │   ├── debug-internal.log
│   │   │   └── debug-core.log -> ~/Library/Caches/wandb/logs/core-debug-*.log
│   │   ├── run-*.wandb
│   │   └── tmp/
│   │       └── code/
│   └── ... (many more runs)
└── README.md
---
