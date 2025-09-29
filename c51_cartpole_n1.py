# train_cartpole_c51.py
import os, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from collections import deque, namedtuple
import wandb
from tqdm import trange

# ----- 0) W&B init -----
wandb.init(
    project="c51-project",
    group="ablation",
    name="c51-cartpole-n1",
    config=dict(
        env_id="CartPole-v1",
        seed=42,
        # C51
        n_atoms=51,
        v_min=0.0,
        v_max=120.0,
        gamma=0.99,

        # Replay and Training
        buffer_size=100_000,
        batch_size=128,
        lr=3e-4,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=25_000,

        # n-step
        n_step=1,
        train_start= 5000, # changed from 1000 to 5000, warmer start to reduce noisy updates
        train_freq=1,

        # target network
        target_update_interval=500,

        # eval and logging
        max_steps=200_000,
        eval_interval=10_000,
        eval_episodes=15,
        snapshot_interval=20_000,
        polyak_tau=0.005,
    ),
)
cfg = wandb.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- 1) Set seed -----
def set_seed(env, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    try:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    except Exception:
        pass


# ----- 2) Create environment -----
env = gym.make(cfg.env_id)
eval_env = gym.make(cfg.env_id)
set_seed(env, cfg.seed); set_seed(eval_env, cfg.seed + 1)
n_actions = env.action_space.n
state_dim = env.observation_space.shape[0]

# ----- 3) Replay Buffer + n step queue -----
Transition = namedtuple("Transition", ["s", "a", "r", "s2", "d"])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def __len__(self):
        return len(self.buf)
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s  = torch.tensor(np.array([t.s  for t in batch]), dtype=torch.float32, device=device)
        a  = torch.tensor(np.array([t.a  for t in batch]), dtype=torch.long,     device=device)
        r  = torch.tensor(np.array([t.r  for t in batch]), dtype=torch.float32, device=device)
        s2 = torch.tensor(np.array([t.s2 for t in batch]), dtype=torch.float32, device=device)
        d  = torch.tensor(np.array([t.d  for t in batch]), dtype=torch.float32, device=device)
        return s, a, r, s2, d

buffer = ReplayBuffer(cfg.buffer_size)
n_step = int(cfg.n_step)
gamma  = float(cfg.gamma)
nq = deque(maxlen=n_step)

def push_n_step(s, a, r, s2, done, buffer):
    nq.append((s, a, r, s2, done))
    if len(nq) < n_step:
        return

    R, (s0, a0, _, _, _ ) = 0.0, nq[0]
    for i, (_, _, r_i, _, d_i) in enumerate(nq):
        R += (gamma ** i) * r_i
        if d_i:  # stop accumulating past terminal
            break
    sN, dN = nq[-1][3], nq[-1][4]
    buffer.push(s0, a0, R, sN, float(dN))

    if dN:   # if terminal, reset the accumulator
        nq.clear()



# ----- 4) C51 Network -----
class C51Net(nn.Module):
    def __init__(self, state_dim, n_actions, n_atoms, v_min, v_max):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms   = n_atoms
        self.v_min     = v_min
        self.v_max     = v_max
        self.delta_z   = (v_max - v_min) / (n_atoms - 1)

        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions * n_atoms),
        )

    def forward(self, x):
        logits = self.net(x)                       # (B, n_actions*n_atoms)
        logits = logits.view(-1, self.n_actions, self.n_atoms)
        log_prob = F.log_softmax(logits, dim=-1)   # per action distribution
        prob = log_prob.exp()
        return prob, log_prob

    def q_values(self, x):
        prob, _ = self.forward(x)
        # expectation over support
        q = torch.sum(prob * self.support, dim=-1)  # (B, n_actions)
        return q

# ----- 5) Online & Target Networks -----

online = C51Net(state_dim, n_actions, cfg.n_atoms, cfg.v_min, cfg.v_max).to(device)
target = C51Net(state_dim, n_actions, cfg.n_atoms, cfg.v_min, cfg.v_max).to(device)
target.load_state_dict(online.state_dict())
optim = torch.optim.Adam(online.parameters(), lr=cfg.lr)

def polyak_update(online, target, tau=0.005):
    with torch.no_grad():
        for p_t, p_o in zip(target.parameters(), online.parameters()):
            p_t.data.mul_(1 - tau).add_(tau * p_o.data)


# ----- 6) Epsilon Greedy Exploration -----
def epsilon_by_step(step):
    eps_start, eps_end, eps_decay = cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_decay_steps
    t = min(1.0, step / eps_decay)
    return eps_start + t * (eps_end - eps_start)

def select_action(state, step):
    eps = epsilon_by_step(step)
    if random.random() < eps:
        return env.action_space.sample(), eps
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = online.q_values(s)                  # (1, n_actions)
        a = int(torch.argmax(q, dim=1).item())
        return a, eps

# ----- 7) Distribution Projection -----

def project_distribution(next_dist, rewards, dones, gamma, v_min, v_max, support):
    # next_dist: (B, n_atoms)  probabilities for chosen next action
    B, n_atoms = next_dist.shape
    delta_z = (v_max - v_min) / (n_atoms - 1)
    # Tz: (B, n_atoms)
    Tz = rewards.unsqueeze(1) + gamma * (1.0 - dones.unsqueeze(1)) * support.unsqueeze(0)
    Tz = torch.clamp(Tz, v_min, v_max)
    b  = (Tz - v_min) / delta_z
    l  = b.floor().long()
    u  = b.ceil().long()

    m = torch.zeros(B, n_atoms, device=next_dist.device)
    # Distribute probability mass
    for i in range(B):
        for j in range(n_atoms):
            lj = l[i, j]
            uj = u[i, j]
            pj = next_dist[i, j]
            if lj == uj:
                m[i, lj] += pj
            else:
                m[i, lj] += pj * (uj - b[i, j])
                m[i, uj] += pj * (b[i, j] - lj)
    return m

# ----- 8) One Training Step to Optimize Loss -----

def train_step():
    if len(buffer) < cfg.train_start:
        return None
    s, a, r, s2, d = buffer.sample(cfg.batch_size)

    with torch.no_grad():
        # online used to select a* (Double-DQN)
        q_next_online = online.q_values(s2)                       # (B, n_actions)
        a_star = torch.argmax(q_next_online, dim=1)               # (B,)
        # target used to evaluate distribution at a*
        next_prob, _ = target.forward(s2)                         # (B, n_actions, n_atoms)
        next_dist = next_prob[torch.arange(cfg.batch_size), a_star] # (B, n_atoms)
        m = project_distribution(
            next_dist, r, d, (cfg.gamma ** cfg.n_step), cfg.v_min, cfg.v_max, target.support
        )  # (B, n_atoms) # changed from cfg.gamma to (cfg.gamma ** cfg.n_step)

    # predicted log-prob for chosen action
    _, log_prob = online.forward(s)                               # (B, n_actions, n_atoms)
    log_prob_a = log_prob[torch.arange(cfg.batch_size), a]        # (B, n_atoms)

    # cross-entropy loss: -sum m * log p
    loss = -(m * log_prob_a).sum(dim=1).mean()

    optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
    optim.step()
    return float(loss.item())


# ----- 9) Distribution Snapshot ----- 

def log_distribution_snapshot(env, step, title="reset_state"):
    # Safe, backend-agnostic plotting; won't crash training
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        o, _ = env.reset(seed=123)  # fixed probe state for comparability
        s = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            prob, _ = online(s)                 # (1, A, Z)
            q = online.q_values(s)              # (1, A)
            a = int(q.argmax(dim=1).item())
            p = prob[0, a].detach().cpu().numpy()
            z = online.support.detach().cpu().numpy()

        fig, ax = plt.subplots()
        ax.stem(z, p)                           # no deprecated kwargs; no unpack
        ax.set_title(f"C51 atoms — {title} — step {step}")
        ax.set_xlabel("Return support"); ax.set_ylabel("Probability")
        wandb.log({"dist_snapshot": wandb.Image(fig), "snapshot_step": step})
        plt.close(fig)
    except Exception as e:
        wandb.log({"snapshot_error": str(e), "snapshot_step": step})


# ----- 10) Checkpointing helper to resume training -----

def find_latest_checkpoint():
    files = glob.glob("checkpoints/c51_step*.pt")
    if not files:
        return None, None
    def step_from_name(path):
        m = re.search(r"c51_step(\d+)\.pt$", os.path.basename(path))
        return int(m.group(1)) if m else -1
    latest = max(files, key=step_from_name)
    step = step_from_name(latest)
    return latest, step

def save_checkpoint(step):
    os.makedirs("checkpoints", exist_ok=True)
    model_path = f"checkpoints/c51_step{step}.pt"
    opt_path   = f"checkpoints/c51_opt_step{step}.pt"
    torch.save(online.state_dict(), model_path)
    torch.save(optim.state_dict(),  opt_path)
    wandb.save(model_path)
    wandb.save(opt_path)

def try_resume():
    latest, step = find_latest_checkpoint()
    if latest is None:
        return 0  # start fresh
    state = torch.load(latest, map_mode=torch.load, map_location=device) if hasattr(torch, "load") else torch.load(latest, map_location=device)
    state = torch.load(latest, map_location=device)
    online.load_state_dict(state)
    target.load_state_dict(online.state_dict())
    opt_path = f"checkpoints/c51_opt_step{step}.pt"
    if os.path.exists(opt_path):
        optim.load_state_dict(torch.load(opt_path, map_location=device))
    wandb.log({"resumed_from_step": step})
    print(f"[RESUME] Loaded {latest} and optimizer ({os.path.exists(opt_path)})")
    return step

# ----- 11) Training Loop -----

os.makedirs("checkpoints", exist_ok=True)

global_step = 0
episode = 0
obs, _ = env.reset()
ep_return = 0.0

pbar = trange(cfg.max_steps, desc="Training", smoothing=0.05)
for _ in pbar:
    a, eps = select_action(obs, global_step)
    next_obs, reward, terminated, truncated, _ = env.step(a)
    done = terminated or truncated

    push_n_step(obs, a, reward, next_obs, done, buffer) # changed from buffer.push(obs, a, reward, next_obs, float(done))
    obs = next_obs
    ep_return += reward
    global_step += 1

    # train
    if global_step % cfg.train_freq == 0:
        loss = train_step()
    else:
        loss = None

    # episode end
    if done:
        wandb.log({"step": global_step, "episode_return": ep_return, "epsilon": eps})
        ep_return = 0.0
        episode += 1
        obs, _ = env.reset() 

    # target updates
    if cfg.polyak_tau:
        polyak_update(online, target, float(cfg.polyak_tau))
    elif global_step % cfg.target_update_interval == 0:
        target.load_state_dict(online.state_dict())

    # log loss
    if loss is not None:
        wandb.log({"step": global_step, "loss": loss, "epsilon": eps})

    # periodic eval
    if cfg.eval_interval and (global_step % cfg.eval_interval == 0):
        def eval_policy(n_episodes=None):
            n_eps = int(n_episodes or cfg.eval_episodes)
            total = 0.0
            for _ in range(n_eps):
                o, _ = eval_env.reset()
                d = False
                while not d:
                    with torch.no_grad():
                        q = online.q_values(torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0))
                        a_eval = int(q.argmax(dim=1).item())
                    o, r, term, trunc, _ = eval_env.step(a_eval)
                    d = term or trunc
                    total += r
            return total / n_eps

        eval_return = eval_policy()
        wandb.log({"step": global_step, "eval_return": eval_return})
        # save checkpoint
        save_checkpoint(global_step)

    # periodic distribution snapshot
    if cfg.snapshot_interval and (global_step % cfg.snapshot_interval == 0):
        log_distribution_snapshot(eval_env, global_step)

    # tick tqdm progress bar
    pbar.update(1)


env.close(); eval_env.close()



