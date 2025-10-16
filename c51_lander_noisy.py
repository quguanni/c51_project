# c51_lander.py
import os, re, glob, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from collections import deque, namedtuple
import wandb
from tqdm import trange

# ---------- 0) W&B ----------
wandb.init(
    project="c51-project",
    group="lander_noisy",
    name="c51-lander_noisy_500kmax_-200300",
    config=dict(
        env_id="LunarLander-v3",
        seed=42,

        # C51 support tuned for LunarLander (~[-200, 300])
        n_atoms=51,          # try 21 / 51 / 101 in ablations
        v_min=0.0,        # try widening/narrowing in support ablations
        v_max=300.0,
        gamma=0.99,

        # Replay & Training
        buffer_size=200_000,
        batch_size=256,
        lr=5e-4,

        # Exploration (ε only used if use_noisy=False)
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=200_000,

        # n-step
        n_step=3,            # sweep {1,3,5} for ablations
        train_start=10_000,
        train_freq=1,

        # Target updates
        target_update_interval=1000,  # ignored if polyak_tau set
        polyak_tau=0.005,             # Polyak averaging on Lander

        # Eval & logging
        max_steps=500_000,
        eval_interval=20_000,
        eval_episodes=20,
        snapshot_interval=40_000,

        # Checkpoints
        ckpt_dir="checkpoints",

        # NoisyNets toggle
        use_noisy=True,              # set False to compare against ε-greedy
        noisy_sigma_init=0.5,        # lower (e.g., 0.3 or 0.1) if exploration too chaotic
    ),
)
cfg = wandb.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 1) Seeding ----------
def set_seed(env, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    try:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    except Exception:
        pass

# ---------- 2) Env ----------
env = gym.make(cfg.env_id)
eval_env = gym.make(cfg.env_id)
set_seed(env, cfg.seed); set_seed(eval_env, cfg.seed + 1)
n_actions = env.action_space.n
state_dim = int(np.prod(env.observation_space.shape))

# ---------- 3) Replay Buffer + n-step ----------
Transition = namedtuple("Transition", ["s", "a", "r", "s2", "d"])
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
        return s, a, r, s2, d

buffer = ReplayBuffer(cfg.buffer_size)
n_step = int(cfg.n_step)
gamma  = float(cfg.gamma)
nq = deque(maxlen=n_step)

def push_n_step(s, a, r, s2, done, buffer):
    nq.append((s, a, r, s2, done))
    if len(nq) < n_step:
        return
    R, (s0, a0, _, _, _) = 0.0, nq[0]
    for i, (_, _, r_i, _, d_i) in enumerate(nq):
        R += (gamma ** i) * r_i
        if d_i:
            break
    sN, dN = nq[-1][3], nq[-1][4]
    buffer.push(s0, a0, R, sN, float(dN))
    if dN:
        nq.clear()

# ---------- 4) NoisyLinear (Factorized) ----------
class NoisyLinear(nn.Module):
    """
    Factorized NoisyNet layer (Fortunato et al., 2018).
    W = μ_w + σ_w ⊙ (f(ε_p) ⊗ f(ε_q)),  b = μ_b + σ_b ⊙ f(ε_p),
    with f(x) = sign(x) * sqrt(|x|).
    """
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Parameters
        self.mu_weight    = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_bias      = nn.Parameter(torch.empty(out_features))
        self.sigma_bias   = nn.Parameter(torch.empty(out_features))
        # Noise buffers
        self.register_buffer("eps_p", torch.zeros(out_features))
        self.register_buffer("eps_q", torch.zeros(in_features))
        self.reset_parameters(sigma_init)

    @staticmethod
    def _f(x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))

    def reset_parameters(self, sigma_init):
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.mu_weight, -bound, bound)
        nn.init.uniform_(self.mu_bias,   -bound, bound)
        nn.init.constant_(self.sigma_weight, sigma_init / math.sqrt(self.in_features))
        nn.init.constant_(self.sigma_bias,   sigma_init / math.sqrt(self.in_features))

    def forward(self, x):
        # sample new noise each forward pass
        self.eps_p = self.eps_p.resize_(self.out_features).normal_()
        self.eps_q = self.eps_q.resize_(self.in_features).normal_()
        f_p = self._f(self.eps_p)
        f_q = self._f(self.eps_q)
        eps_w = torch.ger(f_p, f_q)        # (out,in)
        w = self.mu_weight + self.sigma_weight * eps_w
        b = self.mu_bias   + self.sigma_bias   * f_p
        return F.linear(x, w, b)

# ---------- 5) C51 Network ----------
class C51Net(nn.Module):
    def __init__(self, state_dim, n_actions, n_atoms, v_min, v_max, *, use_noisy=True, noisy_sigma_init=0.5):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms   = n_atoms
        self.v_min     = v_min
        self.v_max     = v_max
        self.delta_z   = (v_max - v_min) / (n_atoms - 1)
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        if use_noisy:
            self.net = nn.Sequential(
                NoisyLinear(state_dim, 128, sigma_init=noisy_sigma_init), nn.ReLU(),
                NoisyLinear(128, 128,       sigma_init=noisy_sigma_init), nn.ReLU(),
                nn.Linear(128, n_actions * n_atoms),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(state_dim, 128), nn.ReLU(),
                nn.Linear(128, 128),       nn.ReLU(),
                nn.Linear(128, n_actions * n_atoms),
            )

    def forward(self, x):
        logits = self.net(x)                        # (B, A*Z)
        logits = logits.view(-1, self.n_actions, self.n_atoms)
        log_prob = F.log_softmax(logits, dim=-1)    # per-action log-probs over atoms
        prob = log_prob.exp()
        return prob, log_prob

    def q_values(self, x):
        prob, _ = self.forward(x)                   # (B, A, Z)
        q = torch.sum(prob * self.support, dim=-1)  # (B, A)
        return q

# ---------- 6) Nets / Optim ----------
online = C51Net(
    state_dim, n_actions, cfg.n_atoms, cfg.v_min, cfg.v_max,
    use_noisy=bool(cfg.use_noisy), noisy_sigma_init=float(cfg.noisy_sigma_init)
).to(device)
target = C51Net(
    state_dim, n_actions, cfg.n_atoms, cfg.v_min, cfg.v_max,
    use_noisy=bool(cfg.use_noisy), noisy_sigma_init=float(cfg.noisy_sigma_init)
).to(device)
target.load_state_dict(online.state_dict())
optim = torch.optim.Adam(online.parameters(), lr=cfg.lr)

def polyak_update(online, target, tau):
    with torch.no_grad():
        for p_t, p_o in zip(target.parameters(), online.parameters()):
            p_t.data.mul_(1 - tau).add_(tau * p_o.data)

# ---------- 7) Action selection ----------
def epsilon_by_step(step):
    t = min(1.0, step / cfg.epsilon_decay_steps)
    return float(cfg.epsilon_start + t * (cfg.epsilon_end - cfg.epsilon_start))

def select_action(state, step):
    # If NoisyNets are enabled, exploration comes from parameter noise → no ε-greedy
    if bool(cfg.use_noisy):
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q = online.q_values(s)
            a = int(torch.argmax(q, dim=1).item())
        return a, 0.0
    # ε-greedy path (for baseline comparisons)
    eps = epsilon_by_step(step)
    if random.random() < eps:
        return env.action_space.sample(), eps
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = online.q_values(s)
        a = int(torch.argmax(q, dim=1).item())
    return a, eps

# ---------- 8) Projection ----------
def project_distribution(next_dist, rewards, dones, gamma_eff, v_min, v_max, support):
    B, n_atoms = next_dist.shape
    delta_z = (v_max - v_min) / (n_atoms - 1)
    Tz = rewards.unsqueeze(1) + gamma_eff * (1.0 - dones.unsqueeze(1)) * support.unsqueeze(0)
    Tz = torch.clamp(Tz, v_min, v_max)
    b  = (Tz - v_min) / delta_z
    l  = b.floor().long(); u = b.ceil().long()
    m = torch.zeros(B, n_atoms, device=next_dist.device)
    for i in range(B):
        for j in range(n_atoms):
            lj, uj, pj = l[i, j], u[i, j], next_dist[i, j]
            if lj == uj:
                m[i, lj] += pj
            else:
                m[i, lj] += pj * (uj - b[i, j])
                m[i, uj] += pj * (b[i, j] - lj)
    return m

# ---------- 9) Train step ----------
def train_step():
    if len(buffer) < cfg.train_start:
        return None
    s, a, r, s2, d = buffer.sample(cfg.batch_size)
    with torch.no_grad():
        q_next_online = online.q_values(s2)
        a_star = torch.argmax(q_next_online, dim=1)
        next_prob, _ = target.forward(s2)
        next_dist = next_prob[torch.arange(cfg.batch_size), a_star]
        m = project_distribution(
            next_dist, r, d, (cfg.gamma ** cfg.n_step),
            cfg.v_min, cfg.v_max, target.support
        )
    _, log_prob = online.forward(s)
    log_prob_a = log_prob[torch.arange(cfg.batch_size), a]
    loss = -(m * log_prob_a).sum(dim=1).mean()
    optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optim.step()
    return float(loss.item())

# ---------- 10) Snapshot (robust) ----------
def log_distribution_snapshot(env_, step, title="reset_state"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        o, _ = env_.reset(seed=123)
        s = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            prob, _ = online(s)
            q = online.q_values(s)
            a = int(q.argmax(dim=1).item())
            p = prob[0, a].detach().cpu().numpy()
            z = online.support.detach().cpu().numpy()
        fig, ax = plt.subplots()
        ax.stem(z, p)
        ax.set_title(f"C51 atoms — {title} — step {step}")
        ax.set_xlabel("Return support"); ax.set_ylabel("Probability")
        wandb.log({"dist_snapshot": wandb.Image(fig), "snapshot_step": step})
        plt.close(fig)
    except Exception as e:
        wandb.log({"snapshot_error": str(e), "snapshot_step": step})

# ---------- 11) Checkpointing (env-scoped + meta) ----------
def ckpt_prefix():
    safe_env = str(cfg.env_id).replace("/", "_")
    noisy_tag = "noisy" if bool(cfg.use_noisy) else "eps"
    return os.path.join(cfg.ckpt_dir, f"{safe_env}_c51_{noisy_tag}_step")

def save_checkpoint(step):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    meta = dict(
        env_id=str(cfg.env_id),
        state_dim=int(state_dim),
        n_actions=int(n_actions),
        n_atoms=int(cfg.n_atoms),
        v_min=float(cfg.v_min),
        v_max=float(cfg.v_max),
        use_noisy=bool(cfg.use_noisy),
    )
    torch.save({"state_dict": online.state_dict(), "meta": meta},
               f"{ckpt_prefix()}{step}.pt")
    torch.save(optim.state_dict(), f"{ckpt_prefix()}{step}_opt.pt")
    wandb.save(f"{ckpt_prefix()}{step}.pt")
    wandb.save(f"{ckpt_prefix()}{step}_opt.pt")

def find_latest_checkpoint():
    files = [f for f in glob.glob(f"{ckpt_prefix()}*.pt") if not f.endswith("_opt.pt")]
    if not files:
        return None, None
    def step_from_name(path):
        m = re.search(r"_step(\d+)\.pt$", os.path.basename(path))
        return int(m.group(1)) if m else -1
    latest = max(files, key=step_from_name)
    return latest, step_from_name(latest)

def try_resume():
    latest, step = find_latest_checkpoint()
    if latest is None:
        return 0
    payload = torch.load(latest, map_location=device)
    if not isinstance(payload, dict) or "state_dict" not in payload or "meta" not in payload:
        print(f"[RESUME] Incompatible checkpoint (no meta): {latest}")
        return 0
    meta = payload["meta"]
    ok = (
        meta.get("env_id") == cfg.env_id and
        int(meta.get("state_dim", -1)) == state_dim and
        int(meta.get("n_actions", -1)) == n_actions and
        int(meta.get("n_atoms", -1)) == int(cfg.n_atoms) and
        float(meta.get("v_min", 0)) == float(cfg.v_min) and
        float(meta.get("v_max", 0)) == float(cfg.v_max) and
        bool(meta.get("use_noisy", False)) == bool(cfg.use_noisy)
    )
    if not ok:
        print(f"[RESUME] Checkpoint meta mismatch; starting fresh.\nMeta: {meta}")
        return 0
    online.load_state_dict(payload["state_dict"])
    target.load_state_dict(online.state_dict())
    opt_path = f"{ckpt_prefix()}{step}_opt.pt"
    if os.path.exists(opt_path):
        optim.load_state_dict(torch.load(opt_path, map_location=device))
    wandb.log({"resumed_from_step": step})
    print(f"[RESUME] Loaded {latest} (step {step}), optimizer={os.path.exists(opt_path)}")
    return step

# ---------- 12) Training loop ----------
os.makedirs(cfg.ckpt_dir, exist_ok=True)
global_step = try_resume()
episode = 0
obs, _ = env.reset()
ep_return = 0.0

remaining = int(cfg.max_steps - global_step)
pbar = trange(remaining, desc="Training", smoothing=0.05)

for _ in pbar:
    a, eps = select_action(obs, global_step)
    next_obs, reward, terminated, truncated, _ = env.step(a)
    done = terminated or truncated

    push_n_step(obs, a, reward, next_obs, done, buffer)
    obs = next_obs
    ep_return += reward
    global_step += 1

    # update
    loss = train_step() if (global_step % cfg.train_freq == 0) else None

    # episode end
    if done:
        wandb.log({"step": global_step, "episode_return": ep_return, "epsilon": eps})
        ep_return = 0.0
        episode += 1
        obs, _ = env.reset()

    # target net
    if cfg.polyak_tau:
        polyak_update(online, target, float(cfg.polyak_tau))
    elif global_step % cfg.target_update_interval == 0:
        target.load_state_dict(online.state_dict())

    # logs
    if loss is not None:
        wandb.log({"step": global_step, "loss": loss, "epsilon": eps})

    # eval + checkpoint
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
        save_checkpoint(global_step)

    if cfg.snapshot_interval and (global_step % cfg.snapshot_interval == 0):
        log_distribution_snapshot(eval_env, global_step)

env.close(); eval_env.close()
