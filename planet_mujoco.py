# =============================================================================
#  planet_mujoco.py  —  PlaNet for MuJoCo continuous-control environments
#
#  Changes from planet.py:
#    • Config gains env_name, action_dim, hidden_size, state_size, enc_dim
#    • ALL model constructors take a cfg argument — no magic numbers in code
#    • Default sizes bumped to hidden=400, state=50 for richer MuJoCo tasks
#    • Drive paths keyed by env_name so Pendulum weights are never overwritten
#    • __main__ targets HalfCheetah-v4 with a virtual display for Colab
#
#  To switch environments, change env_name (and action_dim to match) in Config.
#  Everything else adapts automatically.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
import os
import pathlib
from torch.utils.data import Dataset, DataLoader


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  — single source of truth for every dimension in the model
# ══════════════════════════════════════════════════════════════════════════════
class Config:
    # ── Environment ──────────────────────────────────────────────────────────
    env_name   = 'HalfCheetah-v4'
    #  Common action_dim values:
    #    HalfCheetah-v4 → 6 | Hopper-v4 → 3 | Walker2d-v4 → 6 | Ant-v4 → 8
    action_dim = 6

    # ── Model dimensions ─────────────────────────────────────────────────────
    # Bumped from Pendulum defaults (hidden=200, state=30) because MuJoCo tasks
    # have richer dynamics that require larger representational capacity.
    hidden_size = 400   # GRU / deterministic state h_t
    state_size  = 50    # stochastic latent state s_t
    enc_dim     = 1024  # Encoder output (fixed by the conv stack — do not change
                        # without also changing the conv channel widths)

    # ── Training ─────────────────────────────────────────────────────────────
    total_iterations     = 1000  # MuJoCo tasks need more iterations than Pendulum
    train_steps_per_iter = 100
    batch_size           = 50
    seq_len              = 50    # longer sequences to capture locomotion cycles
    lr                   = 6e-4
    seed_episodes        = 10   # more diverse seeds for harder tasks
    collect_episodes     = 1
    overshoot_d          = 5

    # ── Google Drive paths  (keyed by env_name → no checkpoint conflicts) ────
    drive_base     = '/content/drive/MyDrive/PlaNet'
    episode_dir    = f'{drive_base}/{env_name}/episodes'
    checkpoint_dir = f'{drive_base}/{env_name}/checkpoints'
    viz_dir        = f'{drive_base}/{env_name}/visualizations'
    checkpoint_every  = 10
    keep_checkpoints  = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════════════════
#  1.  ENCODER   (B, 64, 64, 3)  →  (B, enc_dim)
# ══════════════════════════════════════════════════════════════════════════════
class Encoder(nn.Module):
    """
    Four strided convolutions halve spatial resolution each step:
        64×64 → 32×32 → 16×16 → 8×8 → 4×4  (each: kernel=4, stride=2, pad=1)
    A linear layer then projects the 256×4×4 = 4096 features → enc_dim.

    enc_dim (default 1024) is the interface between the encoder and the
    posterior.  Changing it would require updating the Posterior linear as well.
    """
    def __init__(self, cfg: Config, in_channels: int = 3):
        super().__init__()
        self.cv1 = nn.Conv2d(in_channels, 32,  4, 2, 1)   # → (B, 32,  32, 32)
        self.cv2 = nn.Conv2d(32,          64,  4, 2, 1)   # → (B, 64,  16, 16)
        self.cv3 = nn.Conv2d(64,          128, 4, 2, 1)   # → (B, 128,  8,  8)
        self.cv4 = nn.Conv2d(128,         256, 4, 2, 1)   # → (B, 256,  4,  4)
        self.fc  = nn.Linear(256 * 4 * 4, cfg.enc_dim)    # → (B, enc_dim)

    def forward(self, x):
        # x: (B, H, W, C)  —  permute to (B, C, H, W) for Conv2d
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.cv1(x))
        x = F.relu(self.cv2(x))
        x = F.relu(self.cv3(x))
        x = F.relu(self.cv4(x))
        return self.fc(x.flatten(1))   # (B, enc_dim)


# ══════════════════════════════════════════════════════════════════════════════
#  2.  GRU  — deterministic transition
#      h_t  =  GRUCell( [s_{t-1}, a_{t-1}],  h_{t-1} )
# ══════════════════════════════════════════════════════════════════════════════
class GRU(nn.Module):
    """
    Maintains the deterministic hidden state h_t which carries long-range
    temporal information across the sequence.

    Input to GRUCell: concat(s_{t-1}, a_{t-1})
        shape: (B, state_size + action_dim)
    Output h_t: (B, hidden_size)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cell = nn.GRUCell(cfg.state_size + cfg.action_dim, cfg.hidden_size)

    def forward(self, s_t, a_t, h_old):
        # s_t: (B, state_size)  a_t: (B, action_dim)  h_old: (B, hidden_size)
        return self.cell(torch.cat([s_t, a_t], dim=-1), h_old)  # (B, hidden_size)


# ══════════════════════════════════════════════════════════════════════════════
#  3.  PRIOR   p(s_t | h_t)
# ══════════════════════════════════════════════════════════════════════════════
class Prior(nn.Module):
    """
    Predicts the stochastic state from the deterministic state alone.
    Used at planning time (no encoder available) and for the KL loss term.

    h_t (B, hidden_size)  →  mean, std  each (B, state_size)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.fc     = nn.Linear(cfg.hidden_size, 256)
        self.fc_mu  = nn.Linear(256, cfg.state_size)
        self.fc_std = nn.Linear(256, cfg.state_size)

    def forward(self, h):
        x   = F.relu(self.fc(h))
        mu  = self.fc_mu(x)
        std = F.softplus(self.fc_std(x)) + 0.1   # +0.1 floor prevents collapse
        return mu, std


# ══════════════════════════════════════════════════════════════════════════════
#  4.  POSTERIOR   q(s_t | h_t, e_t)
# ══════════════════════════════════════════════════════════════════════════════
class Posterior(nn.Module):
    """
    Refines the prior's state estimate using the real encoded observation.
    Called only during training — the encoder is never run at planning time.

    Input: concat(e_t, h_t)  →  (B, enc_dim + hidden_size)
    Output: mean, std  each (B, state_size)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.fc     = nn.Linear(cfg.enc_dim + cfg.hidden_size, 256)
        self.fc_mu  = nn.Linear(256, cfg.state_size)
        self.fc_std = nn.Linear(256, cfg.state_size)

    def forward(self, e_t, h_t):
        x   = F.relu(self.fc(torch.cat([e_t, h_t], dim=-1)))
        mu  = self.fc_mu(x)
        std = F.softplus(self.fc_std(x)) + 0.1
        return mu, std


# ══════════════════════════════════════════════════════════════════════════════
#  5.  RSSM  — Recurrent State Space Model
# ══════════════════════════════════════════════════════════════════════════════
class RSSM(nn.Module):
    """
    Core PlaNet module combining Encoder + GRU + Prior + Posterior.

    Two operating modes:
      obs_step     — training: uses real obs via encoder → posterior
      imagine_step — planning: prior only, encoder never called
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder   = Encoder(cfg)
        self.gru       = GRU(cfg)
        self.prior     = Prior(cfg)
        self.posterior = Posterior(cfg)

    def obs_step(self, h_old, s_old, obs, a_prev):
        """
        Training step — uses the real observation via the posterior.

        Args:
            h_old : (B, hidden_size)  previous deterministic state
            s_old : (B, state_size)   previous stochastic  state
            obs   : (B, 64, 64, 3)   current pixel obs, normalised [0, 1]
            a_prev: (B, action_dim)   *previous* action a_{t-1}

        Returns:
            p_m, p_s  : prior  params   (B, state_size) each
            q_m, q_s  : posterior params
            h, s      : new (h_t, s_t)
        """
        h          = self.gru(s_old, a_prev, h_old)          # (B, hidden_size)
        e          = self.encoder(obs)                         # (B, enc_dim)
        p_m, p_s   = self.prior(h)                            # (B, state_size) each
        q_m, q_s   = self.posterior(e, h)
        # Reparameterisation: s = mu + std * eps,  eps ~ N(0,I)
        # This lets gradients flow through the sampling op to both networks.
        s          = q_m + q_s * torch.randn_like(q_m)        # (B, state_size)
        return p_m, p_s, q_m, q_s, h, s

    def imagine_step(self, h_old, s_old, a_t):
        """
        Dream/planning step — prior only, NO encoder called.

        Returns: p_m, p_s, h, s
        """
        h          = self.gru(s_old, a_t, h_old)
        p_m, p_s   = self.prior(h)
        s          = p_m + p_s * torch.randn_like(p_m)
        return p_m, p_s, h, s


# ══════════════════════════════════════════════════════════════════════════════
#  6.  DECODER   (h, s)  →  (B, 64, 64, 3)
# ══════════════════════════════════════════════════════════════════════════════
class Decoder(nn.Module):
    """
    Mirrors the Encoder using transposed convolutions.

    concat(h, s)  →  linear  →  reshape  →  4×ConvTranspose2d  →  pixels

    (B, hidden+state)
      → (B, 4096)   linear + relu
      → (B,256,4,4) reshape
      → (B,128,8,8)  ConvTranspose2d
      → (B,64,16,16)
      → (B,32,32,32)
      → (B,3,64,64)  sigmoid → [0,1]
      → (B,64,64,3)  permute back to HWC
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.fc   = nn.Linear(cfg.state_size + cfg.hidden_size, 4096)
        self.dec1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec2 = nn.ConvTranspose2d(128, 64,  4, 2, 1)
        self.dec3 = nn.ConvTranspose2d(64,  32,  4, 2, 1)
        self.dec4 = nn.ConvTranspose2d(32,  3,   4, 2, 1)

    def forward(self, h, s):
        x = F.relu(self.fc(torch.cat([h, s], dim=-1))).reshape(-1, 256, 4, 4)
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        return torch.sigmoid(self.dec4(x)).permute(0, 2, 3, 1)  # (B,64,64,3)


# ══════════════════════════════════════════════════════════════════════════════
#  7.  REWARD HEAD   (h, s)  →  (B, 1)
# ══════════════════════════════════════════════════════════════════════════════
class Reward(nn.Module):
    """
    MLP that predicts the scalar reward from the latent state.
    Training this head keeps the latent space task-relevant, not just visually
    faithful — crucial for the CEM planner to find high-reward trajectories.
    """
    def __init__(self, cfg: Config, hidden_dim: int = 400):
        super().__init__()
        self.fc1 = nn.Linear(cfg.state_size + cfg.hidden_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, h, s):
        x = F.relu(self.fc1(torch.cat([s, h], dim=-1)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)   # (B, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  8.  WORLD MODEL
# ══════════════════════════════════════════════════════════════════════════════
class WorldModel(nn.Module):
    """
    Ties RSSM + Decoder + Reward together and runs them over a (T, B) sequence.

    Also computes the latent overshooting KL (Section 3.3 of the paper):
    for each timestep t, we imagine D steps into the future from the posterior
    state at t and compare those imagined prior distributions to the actual
    posterior distributions at t+1 … t+D.  This forces the prior to be
    accurate not just 1-step ahead (which is all the standard KL covers) but
    across the full planning horizon — making the imagined rollouts reliable
    for CEM.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.rssm        = RSSM(cfg)
        self.decoder     = Decoder(cfg)
        self.reward      = Reward(cfg)
        self.overshoot_d = cfg.overshoot_d
        self.cfg         = cfg   # store for zero-init in forward

    def forward(self, obs_seq, action_seq):
        """
        Args:
            obs_seq    : (T, B, 64, 64, 3)  float32 in [0, 1]
            action_seq : (T, B, action_dim)

        Returns:
            recon_img     : (T, B, 64, 64, 3)
            pred_reward   : (T, B, 1)
            prior_mean/std: (T, B, state_size) each
            post_mean/std : (T, B, state_size) each
            overshoot_kl  : scalar
        """
        T, B   = obs_seq.shape[:2]
        device = obs_seq.device
        cfg    = self.cfg

        # Initialise hidden states to zero — no prior episode context
        h = torch.zeros(B, cfg.hidden_size, device=device)
        s = torch.zeros(B, cfg.state_size,  device=device)

        recon_img, pred_reward = [], []
        prior_mean, prior_std  = [], []
        post_mean,  post_std   = [], []
        h_all, s_all           = [], []

        for t in range(T):
            # Use a_{t-1} to avoid causal leakage (current action unknown)
            prev_a = action_seq[t-1] if t > 0 else torch.zeros_like(action_seq[0])
            p_m, p_s, q_m, q_s, h, s = self.rssm.obs_step(h, s, obs_seq[t], prev_a)

            recon_img.append(self.decoder(h, s))    # (B, 64, 64, 3)
            pred_reward.append(self.reward(h, s))   # (B, 1)
            prior_mean.append(p_m);  prior_std.append(p_s)
            post_mean.append(q_m);   post_std.append(q_s)
            h_all.append(h);         s_all.append(s)

        # ── Latent overshooting ───────────────────────────────────────────────
        os_kl = []
        for t in range(T - 1):
            # Detach so overshooting does not backprop through the main sequence
            hi, si = h_all[t].detach(), s_all[t].detach()
            D = min(self.overshoot_d, T - 1 - t)
            for d in range(1, D + 1):
                im_m, im_s, hi, si = self.rssm.imagine_step(hi, si, action_seq[t+d-1])
                tgt_m = post_mean[t+d].detach()
                tgt_s = post_std[t+d].detach()
                # Closed-form KL divergence between two diagonal Gaussians
                kl = (torch.log(im_s / tgt_s)
                      + (tgt_s**2 + (tgt_m - im_m)**2) / (2 * im_s**2) - 0.5)
                os_kl.append(kl.sum(dim=-1).mean())

        overshoot_kl = (torch.stack(os_kl).mean()
                        if os_kl else torch.tensor(0.0, device=device))

        return (torch.stack(recon_img), torch.stack(pred_reward),
                torch.stack(prior_mean), torch.stack(prior_std),
                torch.stack(post_mean),  torch.stack(post_std),
                overshoot_kl)


# ══════════════════════════════════════════════════════════════════════════════
#  9.  LOSS
# ══════════════════════════════════════════════════════════════════════════════
def calculate_loss(recon_img, img, reward, pred_reward,
                   p_m, p_s, q_m, q_s, overshoot_kl,
                   beta=0.1, beta_overshoot=0.1):
    """
    Three loss terms:
      1. Recon loss — MSE(real_pixels, decoded_pixels), summed over spatial dims
      2. Reward loss — MSE(real_reward, predicted_reward)
      3. KL loss    — D_KL(posterior || prior), weighted by beta
         + overshooting KL weighted by beta_overshoot
    """
    recon_loss = F.mse_loss(img, recon_img, reduction='none').sum(dim=[-1,-2,-3]).mean()
    pred_loss  = F.mse_loss(reward.unsqueeze(-1), pred_reward).mean()
    kl_loss    = (torch.log(p_s / q_s)
                  + (q_s**2 + (q_m - p_m)**2) / (2 * p_s**2) - 0.5).sum(-1).mean()
    return recon_loss + pred_loss + beta * kl_loss + beta_overshoot * overshoot_kl


# ══════════════════════════════════════════════════════════════════════════════
# 10.  PIXEL WRAPPER
# ══════════════════════════════════════════════════════════════════════════════
class PixelWrapper(gym.Wrapper):
    """Replaces the env's proprioceptive obs with rendered 64×64 RGB pixels."""
    def __init__(self, env, render_size: int = 64):
        super().__init__(env)
        self.render_size = render_size
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(render_size, render_size, 3), dtype=np.uint8)

    def _get_pixels(self):
        img = self.env.render()
        if isinstance(img, list):
            img = img[0]
        img = np.asarray(img)
        if img.shape[:2] != (self.render_size, self.render_size):
            img = cv2.resize(img, (self.render_size, self.render_size),
                             interpolation=cv2.INTER_AREA)
        return img   # (H, W, 3) uint8

    def reset(self, **kwargs):
        _ = self.env.reset(**kwargs)
        return self._get_pixels(), {}

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._get_pixels(), reward, terminated, truncated, info


# ══════════════════════════════════════════════════════════════════════════════
# 11.  EPISODE STORAGE
# ══════════════════════════════════════════════════════════════════════════════
class EpisodeStorage:
    """Saves each episode as a compressed .npz on Drive; reloads on restart."""
    def __init__(self, episode_dir: str):
        self.episode_dir   = pathlib.Path(episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.episode_paths = sorted(self.episode_dir.glob('episode_*.npz'))
        print(f"[Storage] {len(self.episode_paths)} existing episodes in {episode_dir}")

    def __len__(self):
        return len(self.episode_paths)

    def add_episode(self, obs_list, action_list, reward_list, terminal_list):
        idx  = len(self.episode_paths)
        path = self.episode_dir / f'episode_{idx:06d}.npz'
        np.savez_compressed(path,
            obs      = np.array(obs_list,      dtype=np.uint8),
            action   = np.array(action_list,   dtype=np.float32),
            reward   = np.array(reward_list,   dtype=np.float32),
            terminal = np.array(terminal_list, dtype=bool))
        self.episode_paths.append(path)


# ══════════════════════════════════════════════════════════════════════════════
# 12.  EPISODE DATASET
# ══════════════════════════════════════════════════════════════════════════════
class EpisodeDataset(Dataset):
    """
    Loads all valid episodes into RAM once, then serves random subsequences.
    Uses a virtual __len__ so the DataLoader always produces full batches.
    """
    def __init__(self, storage: EpisodeStorage, seq_len: int, virtual_len: int):
        self.seq_len, self.virtual_len = seq_len, virtual_len
        self.episodes = []
        skipped = 0
        for path in storage.episode_paths:
            with np.load(path) as ep:
                T = len(ep['reward'])
                if T >= seq_len:
                    self.episodes.append({k: np.array(ep[k]) for k in ep})
                else:
                    skipped += 1
        if skipped:
            print(f"[Dataset] Skipped {skipped} short episodes (< {seq_len} frames)")
        if not self.episodes:
            raise RuntimeError(f"No episodes >= {seq_len} frames. Collect more data.")
        total_mb = sum(e['obs'].nbytes for e in self.episodes) / 1e6
        print(f"[Dataset] {len(self.episodes)} episodes cached ({total_mb:.1f} MB)")

    def __len__(self):
        return self.virtual_len

    def __getitem__(self, _):
        ep    = self.episodes[np.random.randint(len(self.episodes))]
        T     = len(ep['reward'])
        i     = np.random.randint(0, T - self.seq_len + 1)
        obs   = ep['obs'][i:i+self.seq_len].astype(np.float32) / 255.0
        return (torch.from_numpy(obs),
                torch.from_numpy(ep['action'][i:i+self.seq_len]),
                torch.from_numpy(ep['reward'][i:i+self.seq_len]),
                torch.from_numpy(ep['terminal'][i:i+self.seq_len].astype(np.float32)))


def collate_episode_batch(batch):
    """Default collate gives (B,T,…); WorldModel needs (T,B,…) — permute here."""
    obs, action, reward, terminal = zip(*batch)
    return (torch.stack(obs).permute(1,0,2,3,4),
            torch.stack(action).permute(1,0,2),
            torch.stack(reward).permute(1,0),
            torch.stack(terminal).permute(1,0))


# ══════════════════════════════════════════════════════════════════════════════
# 13.  CEM PLANNER
# ══════════════════════════════════════════════════════════════════════════════
class CEMPlanner:
    """
    Cross-Entropy Method planning entirely in latent space.
    Samples num_candidates action sequences, rolls them out via imagine_step,
    scores by summed predicted reward, refits a Gaussian to the top_k elite
    sequences, repeats n_iter times, returns the first action.
    """
    def __init__(self, model: WorldModel, cfg: Config,
                 num_candidates=1000, top_k=100, n_steps=12, n_iter=10):
        self.model          = model
        self.cfg            = cfg
        self.num_candidates = num_candidates
        self.top_k          = top_k
        self.n_steps        = n_steps
        self.n_iter         = n_iter

    @torch.no_grad()
    def plan(self, h, s, device):
        # Gaussian belief over action sequences: (n_steps, action_dim)
        mu  = torch.zeros(self.n_steps, self.cfg.action_dim, device=device)
        std = torch.ones_like(mu)

        for _ in range(self.n_iter):
            # Sample (num_candidates, n_steps, action_dim)
            acts = (mu + std * torch.randn(
                self.num_candidates, self.n_steps,
                self.cfg.action_dim, device=device)).clamp(-1, 1)

            H = h.expand(self.num_candidates, -1)   # (K, hidden_size)
            S = s.expand(self.num_candidates, -1)   # (K, state_size)
            G = torch.zeros(self.num_candidates, device=device)

            for t in range(self.n_steps):
                _, _, H, S = self.model.rssm.imagine_step(H, S, acts[:, t])
                G         += self.model.reward(H, S).squeeze(-1)

            # Refit to top_k elite sequences
            top = G.topk(self.top_k).indices
            mu  = acts[top].mean(0)
            std = acts[top].std(0).clamp(min=1e-4)

        return mu[0]   # first action from the best sequence


# ══════════════════════════════════════════════════════════════════════════════
# 14.  TRAIN STEP
# ══════════════════════════════════════════════════════════════════════════════
def train_step(model, optimizer, obs, action, reward, device):
    """One forward + backward pass.  Clips gradient norm to prevent explosion."""
    recon_image, pred_reward, p_m, p_s, q_m, q_s, os_kl = model(obs, action)
    loss = calculate_loss(recon_image, obs, reward, pred_reward,
                          p_m, p_s, q_m, q_s, os_kl)
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=100.0)
    optimizer.step()
    return loss.item()


# ══════════════════════════════════════════════════════════════════════════════
# 15.  CHECKPOINTING
# ══════════════════════════════════════════════════════════════════════════════
def save_checkpoint(model, optimizer, iteration: int, config: Config):
    ckpt_dir = pathlib.Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f'ckpt_{iteration:06d}.pt'
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration}, path)
    print(f"  [Checkpoint] Saved → {path.name}")
    for old in sorted(ckpt_dir.glob('ckpt_*.pt'))[:-config.keep_checkpoints]:
        old.unlink()


def load_checkpoint(model, optimizer, config: Config) -> int:
    """Returns iteration to resume from (0 = fresh start)."""
    ckpt_dir  = pathlib.Path(config.checkpoint_dir)
    all_ckpts = sorted(ckpt_dir.glob('ckpt_*.pt')) if ckpt_dir.exists() else []
    if not all_ckpts:
        return 0
    data  = torch.load(all_ckpts[-1], map_location=config.device)
    model.load_state_dict(data['model'])
    optimizer.load_state_dict(data['optimizer'])
    start = data['iteration'] + 1
    print(f"[Checkpoint] Resumed from {all_ckpts[-1].name}  (iter {start})")
    return start


# ══════════════════════════════════════════════════════════════════════════════
# 16.  EXPERIENCE COLLECTION
# ══════════════════════════════════════════════════════════════════════════════
def collect_experience(env, model: WorldModel, planner: CEMPlanner,
                       storage: EpisodeStorage, n_episodes: int,
                       config: Config):
    """Run n CEM episodes and persist them to storage."""
    model.eval()
    cfg = config
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        h = torch.zeros(1, cfg.hidden_size, device=cfg.device)
        s = torch.zeros(1, cfg.state_size,  device=cfg.device)
        prev_a = torch.zeros(1, cfg.action_dim, device=cfg.device)
        obs_l, act_l, rew_l, term_l = [], [], [], []

        with torch.no_grad():
            while not done:
                obs_t = (torch.tensor(obs, dtype=torch.float32, device=cfg.device)
                         .unsqueeze(0) / 255.0)                    # (1,64,64,3)
                _, _, _, _, h, s = model.rssm.obs_step(h, s, obs_t, prev_a)
                action = planner.plan(h, s, cfg.device).cpu().numpy()
                prev_a = torch.tensor(action, dtype=torch.float32,
                                      device=cfg.device).reshape(1, -1)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                obs_l.append(obs); act_l.append(action.reshape(-1))
                rew_l.append(float(reward)); term_l.append(bool(done))
                obs = next_obs
        storage.add_episode(obs_l, act_l, rew_l, term_l)
    model.train()


# ══════════════════════════════════════════════════════════════════════════════
# 17.  EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_planner(env, model: WorldModel, planner: CEMPlanner,
                     n_episodes: int, config: Config) -> float:
    """Run n evaluation episodes; returns average reward."""
    model.eval()
    cfg     = config
    rewards = []
    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done   = False
            h = torch.zeros(1, cfg.hidden_size, device=cfg.device)
            s = torch.zeros(1, cfg.state_size,  device=cfg.device)
            prev_a = torch.zeros(1, cfg.action_dim, device=cfg.device)
            ep_r   = 0.0
            while not done:
                obs_t = (torch.tensor(obs, dtype=torch.float32, device=cfg.device)
                         .unsqueeze(0) / 255.0)
                _, _, _, _, h, s = model.rssm.obs_step(h, s, obs_t, prev_a)
                action = planner.plan(h, s, cfg.device).cpu().numpy()
                prev_a = torch.tensor(action, dtype=torch.float32,
                                      device=cfg.device).reshape(1, -1)
                obs, reward, terminated, truncated, _ = env.step(action)
                done  = terminated or truncated
                ep_r += reward
            rewards.append(ep_r)
            print(f"  Eval Episode {ep+1}: {ep_r:.2f}")
    avg = float(np.mean(rewards))
    print(f"  Average: {avg:.2f}")
    model.train()
    return avg


# ══════════════════════════════════════════════════════════════════════════════
# 18.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def train_planner(env, model: WorldModel, optimizer, planner: CEMPlanner,
                  storage: EpisodeStorage, config: Config):
    """
    Main PlaNet training loop.

    Each iteration:
      a. Build DataLoader from all stored episodes (RAM cache)
      b. Run train_steps gradient updates on random subsequences
      c. Collect one new CEM episode → append to storage
      d. Save checkpoint to Drive every checkpoint_every iterations
    """
    cfg        = config
    device     = cfg.device
    start_iter = load_checkpoint(model, optimizer, cfg)

    # ── Seed with random episodes (first run only) ────────────────────────────
    if len(storage) == 0:
        print(f"Seeding with {cfg.seed_episodes} random episodes...")
        for _ in range(cfg.seed_episodes):
            obs, _ = env.reset()
            done   = False
            obs_l, act_l, rew_l, term_l = [], [], [], []
            while not done:
                action   = env.action_space.sample()
                next_obs, r, ter, tru, _ = env.step(action)
                done     = ter or tru
                obs_l.append(obs)
                act_l.append(np.array(action, dtype=np.float32).reshape(-1))
                rew_l.append(float(r))
                term_l.append(bool(done))
                obs = next_obs
            storage.add_episode(obs_l, act_l, rew_l, term_l)
        print(f"  Seeded {len(storage)} episodes.")

    for iteration in range(start_iter, cfg.total_iterations):
        print(f"\nIteration {iteration+1}/{cfg.total_iterations}"
              f"  | Episodes: {len(storage)}")

        virtual_len = cfg.batch_size * cfg.train_steps_per_iter
        dataset = EpisodeDataset(storage, cfg.seq_len, virtual_len)
        loader  = DataLoader(dataset, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=0,
                             collate_fn=collate_episode_batch,
                             pin_memory=False, drop_last=True)

        model.train()
        total_loss, steps = 0.0, 0
        for obs_b, act_b, rew_b, _ in loader:
            if steps >= cfg.train_steps_per_iter:
                break
            total_loss += train_step(model, optimizer,
                                     obs_b.to(device), act_b.to(device),
                                     rew_b.to(device), device)
            steps += 1
        print(f"  Avg Loss: {total_loss/max(steps,1):.4f}  ({steps} steps)")

        collect_experience(env, model, planner, storage, cfg.collect_episodes, cfg)

        if (iteration + 1) % cfg.checkpoint_every == 0:
            save_checkpoint(model, optimizer, iteration, cfg)


# ══════════════════════════════════════════════════════════════════════════════
# 19.  DREAM VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════
def visualize_dream(model: WorldModel, storage: EpisodeStorage, config: Config,
                    episode_idx=0, context_frames=5, dream_steps=50,
                    save_gif=True, save_png=True):
    """
    Open-loop prediction visualizer.
    Context (green) — real obs fed through encoder → posterior warm-up.
    Dream   (red)   — prior only, NO encoder, NO real pixels.
    """
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import imageio

    viz_dir = pathlib.Path(config.viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    ep           = np.load(storage.episode_paths[episode_idx % len(storage)])
    obs_ep       = ep['obs']
    action_ep    = ep['action']
    dream_steps  = max(0, min(dream_steps, len(obs_ep) - context_frames))
    device       = config.device
    cfg          = config
    model.eval()

    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    real_frames, recon_frames, dream_frames_list = [], [], []

    h = torch.zeros(1, cfg.hidden_size, device=device)
    s = torch.zeros(1, cfg.state_size,  device=device)

    with torch.no_grad():
        for t in range(context_frames):
            obs_t  = torch.tensor(obs_ep[t], dtype=torch.float32,
                                  device=device).unsqueeze(0) / 255.0
            prev_a = (torch.tensor(action_ep[t-1], dtype=torch.float32,
                                   device=device).unsqueeze(0)
                      if t > 0 else torch.zeros(1, cfg.action_dim, device=device))
            _, _, _, _, h, s = model.rssm.obs_step(h, s, obs_t, prev_a)
            recon = (model.decoder(h, s)[0].cpu().numpy()*255).clip(0,255).astype(np.uint8)
            real_frames.append(obs_ep[t]); recon_frames.append(recon)

        for d in range(dream_steps):
            t   = context_frames + d
            a_t = (torch.tensor(action_ep[t-1], dtype=torch.float32,
                                device=device).unsqueeze(0)
                   if t < len(action_ep) else torch.zeros(1, cfg.action_dim, device=device))
            _, _, h, s = model.rssm.imagine_step(h, s, a_t)
            dream = (model.decoder(h, s)[0].cpu().numpy()*255).clip(0,255).astype(np.uint8)
            dream_frames_list.append(dream)
            real_frames.append(obs_ep[t] if t < len(obs_ep) else blank)

    recon_all = recon_frames + [blank] * dream_steps
    gif_path = png_path = None

    if save_gif:
        SZ, SW, BH = 256, 10, 22
        sep  = np.ones((SZ, SW, 3), np.uint8) * 200
        labs = ['REAL (ground truth)', 'RECON (posterior)', 'DREAM (prior)']
        frames = []
        for i in range(len(real_frames)):
            is_d = i >= context_frames
            col  = (0,160,0) if not is_d else (180,0,0)
            ps   = [cv2.resize(real_frames[i], (SZ,SZ), interpolation=cv2.INTER_NEAREST),
                    cv2.resize(recon_all[i],   (SZ,SZ), interpolation=cv2.INTER_NEAREST),
                    cv2.resize(dream_frames_list[i-context_frames] if is_d else blank,
                               (SZ,SZ), interpolation=cv2.INTER_NEAREST)]
            for pi, p in enumerate(ps):
                p[:BH] = col
                cv2.putText(p, labs[pi], (4,BH-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(ps[0], f'{"CONTEXT" if not is_d else "DREAM"}  t={i}',
                        (4,BH+18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,0), 1, cv2.LINE_AA)
            frames.append(np.hstack([ps[0], sep, ps[1], sep, ps[2]]))
        gif_path = viz_dir / f'dream_ep{episode_idx:04d}.gif'
        imageio.mimsave(str(gif_path), frames, fps=5, loop=0)
        print(f"[Viz] GIF → {gif_path}")

    if save_png:
        N=5; ctx=list(range(context_frames))
        early=list(range(context_frames, context_frames+min(N,dream_steps)))
        ls=context_frames+dream_steps-min(N,dream_steps)
        late=list(range(ls, context_frames+dream_steps))
        gap=(context_frames+N)<ls
        cols=ctx+early+(['gap'] if gap else [])+late; nc=len(cols)
        fig=plt.figure(figsize=(1.6+nc*3, 3*3+1.8), facecolor='#181818')
        gs=fig.add_gridspec(3,nc, left=1.6/(1.6+nc*3), right=0.99,
                            top=0.87, bottom=0.06, wspace=0.07, hspace=0.22)
        RM=[('Real\nGround Truth','#43a047'),
            ('Recon\nPosterior q(s|h,e)','#1e88e5'),
            ('Dream\nPrior p(s|h) only','#e53935')]
        for ri,(rl,rc) in enumerate(RM):
            for ci,ti in enumerate(cols):
                ax=fig.add_subplot(gs[ri,ci])
                ax.set_facecolor('#181818'); ax.set_xticks([]); ax.set_yticks([])
                if ti=='gap':
                    ax.axis('off')
                    ax.text(0.5,0.5,'···\nskipped',ha='center',va='center',
                            fontsize=13,color='#888',transform=ax.transAxes,fontstyle='italic')
                    if ri==0: ax.set_title('···',fontsize=12,color='#666',pad=7)
                    continue
                ic=ti<context_frames; bc='#66bb6a' if ic else '#ef5350'
                if   ri==0: img=real_frames[ti]
                elif ri==1: img=recon_all[ti] if ic else blank
                else:       img=dream_frames_list[ti-context_frames] if not ic else blank
                ax.imshow(img, interpolation='nearest', aspect='equal')
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_edgecolor(bc); sp.set_linewidth(3.5)
                if ri==0:
                    ax.set_title(f't={ti}\n[{"CTX" if ic else "DREAM"}]',
                                 fontsize=12,pad=6,color=bc,fontweight='bold')
            yf=1.0-(ri+0.5)/3.0
            fig.text(0.005, 0.06+yf*0.81, rl, ha='left', va='center',
                     fontsize=12, color=rc, fontweight='bold',
                     rotation=90, rotation_mode='anchor')
        fig.suptitle(f'Dream  ·  {config.env_name}  ·  Ep {episode_idx}  ·  '
                     f'{context_frames} context + {dream_steps} dream',
                     fontsize=14, color='white', y=0.97)
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(facecolor='#66bb6a',label='Context'),
                             Patch(facecolor='#ef5350',label='Dream')],
                   loc='lower center', ncol=2, fontsize=11,
                   facecolor='#2a2a2a', edgecolor='#555', labelcolor='white',
                   bbox_to_anchor=(0.5,0.0))
        png_path=viz_dir/f'dream_grid_ep{episode_idx:04d}.png'
        fig.savefig(str(png_path), dpi=180, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[Viz] PNG → {png_path}")
    return gif_path, png_path


# ══════════════════════════════════════════════════════════════════════════════
# 20.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # ── Colab: start a virtual display for MuJoCo rendering ──────────────────
    try:
        from pyvirtualdisplay import Display
        Display(visible=0, size=(1400, 900)).start()
        print('[Display] Virtual display started')
    except ImportError:
        print('[Display] pyvirtualdisplay not found — assuming local display')

    # ── Mount Google Drive (Colab only) ───────────────────────────────────────
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        print('[Drive] Mounted')
    except ImportError:
        Config.drive_base     = './local_storage'
        Config.episode_dir    = f'{Config.drive_base}/{Config.env_name}/episodes'
        Config.checkpoint_dir = f'{Config.drive_base}/{Config.env_name}/checkpoints'
        Config.viz_dir        = f'{Config.drive_base}/{Config.env_name}/visualizations'

    config = Config()
    device = config.device
    print(f'[Config] Device: {device}  |  Env: {config.env_name}'
          f'  |  action_dim={config.action_dim}'
          f'  |  hidden={config.hidden_size}  state={config.state_size}')

    # ── Environment ───────────────────────────────────────────────────────────
    base_env = gym.make(config.env_name, render_mode='rgb_array')
    env      = PixelWrapper(base_env, render_size=64)

    # ── Model — all dims driven by cfg, no magic numbers ─────────────────────
    model     = WorldModel(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    planner   = CEMPlanner(model, config)
    storage   = EpisodeStorage(config.episode_dir)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'[Model] {total_params:,} parameters')

    # ── Train ─────────────────────────────────────────────────────────────────
    train_planner(env, model, optimizer, planner, storage, config)

    # ── Evaluate & checkpoint ─────────────────────────────────────────────────
    evaluate_planner(env, model, planner, 5, config)
    save_checkpoint(model, optimizer, config.total_iterations - 1, config)

    # ── Visualize ─────────────────────────────────────────────────────────────
    for ep_idx in range(min(3, len(storage))):
        visualize_dream(model, storage, config,
                        episode_idx=ep_idx, context_frames=5, dream_steps=50)
    print('Done. Check Drive at:', config.viz_dir)
