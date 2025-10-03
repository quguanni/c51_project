# Distributional RL Repro (baseline + scaffold)

Goal: a clean, reproducible baseline (DQN) with seeds/configs/logging and slots for **C51** and **QR-DQN** ablations (support bounds, atoms, n-step).

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python dqn_cartpole.py 
python dqn_lander.py
python c51_cartpole.py
python c51_lander.py
