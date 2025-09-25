# train_cartpole_dqn.py
import os, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym
from collections import deque, namedtuple
from tqdm import trange
import wandb

# ===== W&B =====
wandb.init(
    project="c51-sprint",
    group="ablation",
    name="dqn-cartpole",
    config=dict(
        env_id="CartPole-v1",
        seed=42,
        gamma=0.99,
        buffer_size=100_000,
        batch_size=128,
        lr=3e-4,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=25_000,
        target_update_interval=500,
        train_start=5_000, # changed from 1_000 to 5_000
        train_freq=1,
        max_steps=200_000,
        eval_interval=10_000,
        double_dqn=True,  # keep True for standard modern baseline
    ),
)
cfg = wandb.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Seeding =====
def set_seed(env, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    try: env.reset(seed=seed); env.action_space.seed(seed)
    except: pass

# ===== Env =====
env = gym.make(cfg.env_id)
eval_env = gym.make(cfg.env_id)
set_seed(env, cfg.seed); set_seed(eval_env, cfg.seed+1)
state_dim = env.observation_space.shape[0]
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

buffer = ReplayBuffer(cfg.buffer_size)

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
    if len(buffer) < cfg.train_start: return None
    s,a,r,s2,d = buffer.sample(cfg.batch_size)

    with torch.no_grad():
        if cfg.double_dqn:
            # a* from online, value from target (Double DQN)
            a_star = online(s2).argmax(dim=1)                       # (B,)
            q_next = target(s2).gather(1, a_star.unsqueeze(1)).squeeze(1)  # (B,)
        else:
            q_next = target(s2).max(dim=1).values                   # (B,)
        target_q = r + (1.0 - d) * cfg.gamma * q_next               # (B,)

    q = online(s).gather(1, a.unsqueeze(1)).squeeze(1)              # (B,)
    loss = F.smooth_l1_loss(q, target_q)   # Huber loss

    optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optim.step()
    return float(loss.item())

# ===== Eval =====
def eval_policy(n_episodes=15): # changed from 5 to 15
    total = 0.0
    for _ in range(n_episodes):
        o,_ = eval_env.reset()
        done = False
        while not done:
            with torch.no_grad():
                q = online(torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0))
                a = int(q.argmax(dim=1))
            o,r,term,trunc,_ = eval_env.step(a)
            done = term or trunc
            total += r
    return total / n_episodes

# ===== Loop =====
os.makedirs("checkpoints", exist_ok=True)
global_step, episode, ep_return = 0, 0, 0.0
obs,_ = env.reset()
from math import inf
best_eval = -inf

for _ in trange(cfg.max_steps, desc="DQN"):
    a, eps = select_action(obs, global_step)
    next_obs, reward, terminated, truncated, _ = env.step(a)
    done = terminated or truncated
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
    if global_step % cfg.target_update_interval == 0:
        target.load_state_dict(online.state_dict())

    # periodic eval + checkpoint
    if cfg.eval_interval and (global_step % cfg.eval_interval == 0):
        eval_ret = eval_policy()
        wandb.log({"step": global_step, "eval_return": eval_ret})
        torch.save(online.state_dict(), f"checkpoints/dqn_step{global_step}.pt")
        wandb.save(f"checkpoints/dqn_step{global_step}.pt")
