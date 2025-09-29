# train_cartpole_dqn.py
import os, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym
from collections import deque, namedtuple
from tqdm import trange
import wandb
from math import inf

# ===== W&B =====
wandb.init(
    project="c51-project",
    group="lander",
    name="dqn-lander-1msteps",
    config=dict(
        env_id="LunarLander-v3",
        seed=42,
        gamma=0.99,
        # Replay / Optimization
        buffer_size=200_000,
        batch_size=256,
        lr=3e-4,

        # Epsilon-greedy Exploration
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=200_000,

        # Training schedule and target network updates
        train_start=10_000, # changed from 1_000 to 5_000
        train_freq=1,
        target_update_interval=500,
        max_steps=1_000_000,

        # Evaluation
        eval_interval=20_000,

        # Double DQN
        double_dqn=True,  # keep True for standard modern baseline
    ),
)
cfg = wandb.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Safe getters with defaults (robust to missing config keys)
GAMMA                  = float(getattr(cfg, "gamma", 0.99))
BUFFER_SIZE            = int(getattr(cfg, "buffer_size", 200_000))
BATCH_SIZE             = int(getattr(cfg, "batch_size", 256))
LR                     = float(getattr(cfg, "lr", 3e-4))
EPS_START              = float(getattr(cfg, "epsilon_start", 1.0))
EPS_END                = float(getattr(cfg, "epsilon_end", 0.05))
EPS_DECAY_STEPS        = int(getattr(cfg, "epsilon_decay_steps", 200_000))
TRAIN_START            = int(getattr(cfg, "train_start", 10_000))
TRAIN_FREQ             = int(getattr(cfg, "train_freq", 1))
TARGET_UPDATE_INTERVAL = int(getattr(cfg, "target_update_interval", 1_000))
MAX_STEPS              = int(getattr(cfg, "max_steps", 1_000_000))
EVAL_INTERVAL          = int(getattr(cfg, "eval_interval", 20_000))
EVAL_EPISODES          = int(getattr(cfg, "eval_episodes", 20))
DOUBLE_DQN             = bool(getattr(cfg, "double_dqn", True))

# ===== Seeding =====
def set_seed(env, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    try: 
        env.reset(seed=seed); 
        env.action_space.seed(seed)
    except Exception: 
        pass

# ===== Env =====
env = gym.make(getattr(cfg, "env_id", "LunarLander-v2"))
eval_env = gym.make(getattr(cfg, "env_id", "LunarLander-v2"))
set_seed(env, int(getattr(cfg, "seed", 42)))
set_seed(eval_env, int(getattr(cfg, "seed", 42)) + 1)

# Flatten obs just in case shape changes later
state_dim = int(np.prod(env.observation_space.shape))
n_actions = env.action_space.n

# ===== Replay Buffer =====
Transition = namedtuple("Transition", ["s","a","r","s2","d"])
class ReplayBuffer:
    def __init__(self, capacity): self.buf = deque(maxlen=capacity)
    def push(self, *args): self.buf.append(Transition(*args))
    def __len__(self): return len(self.buf)
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s  = torch.tensor(np.array([t.s  for t in batch]), dtype=torch.float32, device=device)
        a  = torch.tensor(np.array([t.a  for t in batch]), dtype=torch.long,     device=device)
        r  = torch.tensor(np.array([t.r  for t in batch]), dtype=torch.float32, device=device)
        s2 = torch.tensor(np.array([t.s2 for t in batch]), dtype=torch.float32, device=device)
        d  = torch.tensor(np.array([t.d  for t in batch]), dtype=torch.float32, device=device)
        return s,a,r,s2,d

buffer = ReplayBuffer(BUFFER_SIZE)

# ===== DQN Network (same hidden sizes as your C51 MLP) =====
class QNet(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )
    def forward(self, x): return self.net(x)  # (B, n_actions)

online = QNet(state_dim, n_actions).to(device)
target = QNet(state_dim, n_actions).to(device)
target.load_state_dict(online.state_dict())
optim = torch.optim.Adam(online.parameters(), lr=cfg.lr)

# ===== ε-greedy policy =====
def epsilon_by_step(step):
    t = min(1.0, step / cfg.epsilon_decay_steps)
    return cfg.epsilon_start + t * (cfg.epsilon_end - cfg.epsilon_start)

def select_action(state, step):
    eps = epsilon_by_step(step)
    if random.random() < eps:
        return env.action_space.sample(), eps
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = online(s)
        return int(q.argmax(dim=1)), eps

# ===== One training step =====
def train_step():
    if len(buffer) < TRAIN_START: return None
    s,a,r,s2,d = buffer.sample(BATCH_SIZE)

    with torch.no_grad():
        if DOUBLE_DQN:
            # a* from online, value from target (Double DQN)
            a_star = online(s2).argmax(dim=1)                       # (B,)
            q_next = target(s2).gather(1, a_star.unsqueeze(1)).squeeze(1)  # (B,)
        else:
            q_next = target(s2).max(dim=1).values                   # (B,)
        target_q = r + (1.0 - d) * GAMMA * q_next               # (B,)

    q = online(s).gather(1, a.unsqueeze(1)).squeeze(1)              # (B,)
    loss = F.smooth_l1_loss(q, target_q)   # Huber loss

    optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optim.step()
    return float(loss.item())

# ===== Eval =====
def eval_policy(n_episodes=None): 
    n_eps = int(n_episodes or EVAL_EPISODES)
    total = 0.0
    for _ in range(n_eps):
        o,_ = eval_env.reset()
        done = False
        while not done:
            with torch.no_grad():
                q = online(torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0))
                a = int(q.argmax(dim=1).item())
            o,r,term,trunc,_ = eval_env.step(a)
            done = term or trunc
            total += r
    return total / n_eps

# ===== Loop =====
os.makedirs("checkpoints", exist_ok=True)
global_step, episode, ep_return = 0, 0, 0.0
obs,_ = env.reset()
best_eval = float("-inf")

pbar = trange(MAX_STEPS, desc="DQN", smoothing=0.05)
for _ in pbar:
    # Action selection
    a, eps = select_action(obs, global_step)
    next_obs, reward, terminated, truncated, _ = env.step(a)
    done = terminated or truncated

    # Store transition
    buffer.push(obs, a, reward, next_obs, float(done))
    obs = next_obs
    ep_return += reward
    global_step += 1

    # train
    loss = train_step()
    if loss is not None:
        wandb.log({"step": global_step, "loss": loss, "epsilon": eps})

    # episode end
    if done:
        wandb.log({"step": global_step, "episode_return": ep_return})
        ep_return = 0.0; episode += 1
        obs,_ = env.reset()

    # target sync
    if global_step % TARGET_UPDATE_INTERVAL == 0:
        target.load_state_dict(online.state_dict())

    # periodic eval + checkpoint
    if EVAL_INTERVAL and (global_step % EVAL_INTERVAL == 0):
        eval_ret = eval_policy()
        best_eval = max(best_eval, eval_ret)
        wandb.log({"step": global_step, "eval_return": eval_ret, "best_eval": best_eval})
        ckpt = f"checkpoints/dqn_step{global_step}.pt"
        torch.save(online.state_dict(), ckpt)
        wandb.save(ckpt)

env.close(); eval_env.close()
