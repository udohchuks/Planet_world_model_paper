
# pip install imageio imageio-ffmpeg matplotlib
# =============================================================================
#  PlaNet – Planning with Latent Dynamics  (single-file implementation)
#  Paper: "Learning Latent Dynamics for Planning from Pixels" (Hafner et al.)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np

# pip install gymnasium
import gymnasium as gym
from gymnasium import spaces
import cv2

# ── NEW IMPORTS for Dataset / DataLoader / Drive persistence ─────────────────
import os
import pathlib
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────────────────────
#  1. ENCODER  (observation → embedding)
#     Input : (B, 64, 64, 3)   in HWC format
#     Output: (B, 1024)
# ──────────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # Spatial sizes after each conv (stride=2, kernel=4, pad=1):
        # 64→32  32→16  16→8  8→4
        self.cv1 = nn.Conv2d(in_channels, 32,  4, 2, 1)  # → 32×32
        self.cv2 = nn.Conv2d(32,          64,  4, 2, 1)  # → 16×16
        self.cv3 = nn.Conv2d(64,          128, 4, 2, 1)  # → 8×8
        self.cv4 = nn.Conv2d(128,         256, 4, 2, 1)  # → 4×4
        self.fn  = nn.Linear(256 * 4 * 4, 1024)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)          # (B, H, W, C) → (B, C, H, W)
        x = F.relu(self.cv1(x))
        x = F.relu(self.cv2(x))
        x = F.relu(self.cv3(x))
        x = F.relu(self.cv4(x))
        x = x.reshape(x.size(0), -1)        # flatten
        return self.fn(x)                   # (B, 1024)


# ──────────────────────────────────────────────────────────────────────────────
#  2. GRU  – deterministic recurrent state
#     h_t = GRUCell(concat(s_{t-1}, a_{t-1}), h_{t-1})
# ──────────────────────────────────────────────────────────────────────────────
class GRU(nn.Module):
    def __init__(self, state_size=30, action_dim=1, hidden_size=200):
        super().__init__()
        self.gru = nn.GRUCell(state_size + action_dim, hidden_size)

    def forward(self, s_t, a_t, h_old):
        x = torch.cat([s_t, a_t], dim=-1)  # (B, state+action)
        return self.gru(x, h_old)           # (B, hidden)


# ──────────────────────────────────────────────────────────────────────────────
#  3. POSTERIOR  q(s_t | h_t, e_t)
#     Input : h_t (B,200), e_t (B,1024)
#     Output: (mean, std) each (B, 30)
# ──────────────────────────────────────────────────────────────────────────────
class Posterior(nn.Module):
    def __init__(self, output=30):
        super().__init__()
        self.fc     = nn.Linear(1024 + 200, 256)
        self.fc_mu  = nn.Linear(256, output)
        self.fc_std = nn.Linear(256, output)

    def forward(self, e_t, h_t):
        eps = 0.1                                   # minimum std for stability
        x   = torch.cat([e_t, h_t], dim=-1)
        x   = F.relu(self.fc(x))
        mean = self.fc_mu(x)
        std  = F.softplus(self.fc_std(x)) + eps
        return mean, std


# ──────────────────────────────────────────────────────────────────────────────
#  4. PRIOR  p(s_t | h_t)
#     Input : h_t (B,200)
#     Output: (mean, std) each (B, 30)
# ──────────────────────────────────────────────────────────────────────────────
class Prior(nn.Module):
    def __init__(self, hidden_size=200, output_size=30):
        super().__init__()
        self.fc     = nn.Linear(hidden_size, 256)
        self.fc_mu  = nn.Linear(256, output_size)
        self.fc_std = nn.Linear(256, output_size)

    def forward(self, h):
        eps = 0.1
        x   = F.relu(self.fc(h))
        mu  = self.fc_mu(x)
        std = F.softplus(self.fc_std(x)) + eps
        return mu, std


# ──────────────────────────────────────────────────────────────────────────────
#  5. RSSM  – Recurrent State Space Model
#     obs_step    : uses real observation  (training)
#     imagine_step: prior only             (planning / overshooting)
# ──────────────────────────────────────────────────────────────────────────────
class RSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder   = Encoder()
        self.gru       = GRU()
        self.posterior = Posterior()
        self.prior     = Prior()

    def obs_step(self, h_old, s_old, obs, a_t):
        """One step with a real observation.

        Args:
            h_old: (B, 200) previous deterministic state
            s_old: (B, 30)  previous stochastic state
            obs  : (B, 64, 64, 3) current pixel observation (normalised 0–1)
            a_t  : (B, action_dim) **previous** action (a_{t-1})

        Returns:
            m_pr, std_pr : prior  distribution parameters
            m_po, std_po : posterior distribution parameters
            h, s         : new deterministic and stochastic states
        """
        h              = self.gru(s_old, a_t, h_old)
        e              = self.encoder(obs)
        m_pr, std_pr   = self.prior(h)
        m_po, std_po   = self.posterior(e, h)
        s              = m_po + std_po * torch.randn_like(m_po)   # reparameterise
        return m_pr, std_pr, m_po, std_po, h, s

    def imagine_step(self, h_old, s_old, a_t):
        """One step using only the prior (no observation)."""
        h            = self.gru(s_old, a_t, h_old)
        m_pr, std_pr = self.prior(h)
        s            = m_pr + std_pr * torch.randn_like(m_pr)
        return m_pr, std_pr, h, s


# ──────────────────────────────────────────────────────────────────────────────
#  6. DECODER  (h, s → reconstructed image)
#     Output shape: (B, 64, 64, 3) in HWC format, values in [0, 1]
# ──────────────────────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, state_size=30, hidden_size=200):
        super().__init__()
        self.fc1  = nn.Linear(state_size + hidden_size, 4096)     # → 256×4×4
        self.dec1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)         # → 128×8×8
        self.dec2 = nn.ConvTranspose2d(128, 64,  4, 2, 1)         # → 64×16×16
        self.dec3 = nn.ConvTranspose2d(64,  32,  4, 2, 1)         # → 32×32×32
        self.dec4 = nn.ConvTranspose2d(32,  3,   4, 2, 1)         # → 3×64×64

    def forward(self, h, s):
        x   = torch.cat([h, s], dim=-1)
        x   = F.relu(self.fc1(x))
        x   = x.reshape(-1, 256, 4, 4)
        x   = F.relu(self.dec1(x))
        x   = F.relu(self.dec2(x))
        x   = F.relu(self.dec3(x))
        x   = torch.sigmoid(self.dec4(x))
        obs = x.permute(0, 2, 3, 1)        # (B, C, H, W) → (B, H, W, C)
        return obs


# ──────────────────────────────────────────────────────────────────────────────
#  7. REWARD  predictor  (h, s → scalar reward)
# ──────────────────────────────────────────────────────────────────────────────
class Reward(nn.Module):
    def __init__(self, s_size=30, hidden_size=200, hidden_dim=400):
        super().__init__()
        self.fc1 = nn.Linear(s_size + hidden_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, h, s):
        x = torch.cat([s, h], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ──────────────────────────────────────────────────────────────────────────────
#  8. WORLD MODEL  (ties everything together & computes latent overshooting)
# ──────────────────────────────────────────────────────────────────────────────
class WorldModel(nn.Module):
    def __init__(self, overshoot_d=5):
        super().__init__()
        self.rssm        = RSSM()
        self.decoder     = Decoder()
        self.reward      = Reward()
        self.overshoot_d = overshoot_d

    def forward(self, obs_seq, action_seq):
        """
        Args:
            obs_seq   : (T, B, 64, 64, 3)  pixel observations, normalised [0,1]
            action_seq: (T, B, action_dim)  actions

        Returns:
            recon_img     : (T, B, 64, 64, 3)
            pred_reward   : (T, B, 1)
            prior_mean/std: (T, B, 30)
            post_mean/std : (T, B, 30)
            overshoot_kl  : scalar
        """
        T, B   = obs_seq.shape[:2]
        device = obs_seq.device

        h = torch.zeros(B, 200, device=device)
        s = torch.zeros(B, 30,  device=device)

        recon_img, pred_reward               = [], []
        prior_mean, prior_std                = [], []
        post_mean,  post_std                 = [], []
        h_all, s_all                         = [], []

        for t in range(T):
            # Fix 4 – use a_{t-1} not a_t (prevents causal leakage)
            prev_action = action_seq[t - 1] if t > 0 else torch.zeros_like(action_seq[0])
            p_m, p_s, q_m, q_s, h, s = self.rssm.obs_step(h, s, obs_seq[t], prev_action)

            recon_img.append(self.decoder(h, s))
            pred_reward.append(self.reward(h, s))

            prior_mean.append(p_m);  prior_std.append(p_s)
            post_mean.append(q_m);   post_std.append(q_s)
            h_all.append(h);         s_all.append(s)

        # ── Latent overshooting ──────────────────────────────────────────
        # KL( imagined prior  ‖  posterior target )
        overshoot_kl_terms = []
        for t in range(T - 1):
            h_im = h_all[t].detach()
            s_im = s_all[t].detach()
            D    = min(self.overshoot_d, T - 1 - t)

            for d in range(1, D + 1):
                im_m, im_s, h_im, s_im = self.rssm.imagine_step(h_im, s_im, action_seq[t + d - 1])
                target_m = post_mean[t + d].detach()
                target_s = post_std[t + d].detach()

                # KL( N(target) ‖ N(imagined) )
                kl = (
                    torch.log(im_s / target_s)
                    + (target_s ** 2 + (target_m - im_m) ** 2) / (2 * im_s ** 2)
                    - 0.5
                )
                overshoot_kl_terms.append(kl.sum(dim=-1).mean())

        if overshoot_kl_terms:
            overshoot_kl = torch.stack(overshoot_kl_terms).mean()
        else:
            overshoot_kl = torch.tensor(0.0, device=device)

        return (
            torch.stack(recon_img),
            torch.stack(pred_reward),
            torch.stack(prior_mean),
            torch.stack(prior_std),
            torch.stack(post_mean),
            torch.stack(post_std),
            overshoot_kl,
        )


# ──────────────────────────────────────────────────────────────────────────────
#  9. LOSS  – reconstruction + reward + KL + overshoot KL
#     KL formula: D_KL( Q ‖ P ) = log(σ_p/σ_q) + (σ_q²+(μ_q-μ_p)²)/(2σ_p²) - ½
# ──────────────────────────────────────────────────────────────────────────────
def calculate_loss(recon_img, img, reward, pred_reward,
                   p_m, p_s, q_m, q_s, overshoot_kl,
                   beta=0.1, beta_overshoot=0.1):
    # Image reconstruction (sum over pixels, mean over batch+time)
    recon_loss = F.mse_loss(img, recon_img, reduction='none').sum(dim=[-1, -2, -3]).mean()

    # Reward prediction
    pred_loss = F.mse_loss(reward.unsqueeze(-1), pred_reward, reduction='none').mean()

    # KL divergence  D_KL( posterior ‖ prior )
    kl_loss = torch.log(p_s / q_s) + ((q_s ** 2 + (q_m - p_m) ** 2) / (2 * p_s ** 2)) - 0.5
    kl_loss = kl_loss.sum(dim=-1).mean()

    return recon_loss + pred_loss + beta * kl_loss + beta_overshoot * overshoot_kl


# ──────────────────────────────────────────────────────────────────────────────
# 10a. EPISODE STORAGE  [NEW – replaces in-memory circular buffer]
#
# WHAT:  EpisodeStorage saves each collected episode as a compressed .npz file
#        directly to Google Drive (or any folder you point it at).
#
# WHY:   Colab VMs are ephemeral – their RAM and /tmp vanish when the session
#        ends.  By writing raw episode files to Drive we get:
#          • Free persistent storage that survives runtime restarts.
#          • We can inspect / visualise individual episodes in Drive.
#          • The dataset can grow across many sessions without re-collecting.
#
# HOW:   On every call to add_episode() we:
#          1. Stack the transition lists into numpy arrays.
#          2. Save them as episode_XXXXXX.npz in the given directory.
#          3. Append the filename to a manifest list so EpisodeDataset can
#             index them without scanning the folder on every access.
# ──────────────────────────────────────────────────────────────────────────────
class EpisodeStorage:
    def __init__(self, episode_dir: str):
        self.episode_dir = pathlib.Path(episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=True)  # create Drive folder if missing
        # Keep an ordered list of all episode file paths for fast indexing
        self.episode_paths: list[pathlib.Path] = sorted(self.episode_dir.glob('episode_*.npz'))
        print(f"[EpisodeStorage] Found {len(self.episode_paths)} existing episodes in {self.episode_dir}")

    # ── Add a single complete episode ────────────────────────────────────────
    def add_episode(self, obs_list, action_list, reward_list, terminal_list):
        """
        Args:
            obs_list     : list of np.uint8 arrays  (T, 64, 64, 3)
            action_list  : list of np.float32 arrays (T, action_dim)
            reward_list  : list of floats
            terminal_list: list of bools
        """
        idx  = len(self.episode_paths)           # unique episode index
        path = self.episode_dir / f'episode_{idx:06d}.npz'

        np.savez_compressed(
            path,
            obs      = np.array(obs_list,      dtype=np.uint8),    # (T, 64, 64, 3)
            action   = np.array(action_list,   dtype=np.float32),  # (T, action_dim)
            reward   = np.array(reward_list,   dtype=np.float32),  # (T,)
            terminal = np.array(terminal_list, dtype=bool),        # (T,)
        )
        self.episode_paths.append(path)

    def __len__(self):
        return len(self.episode_paths)


# ──────────────────────────────────────────────────────────────────────────────
# 10b. EPISODE DATASET  [NEW – virtual-length design]
#
# WHAT:  A PyTorch Dataset that wraps the saved .npz files on Drive and
#        samples random fixed-length subsequences for BPTT training.
#
# WHY:   Using Dataset + DataLoader gives us:
#          • Multi-worker prefetching (num_workers > 0) so GPU never starves.
#          • Standard PyTorch shuffle / sampler interface.
#          • Lazy loading: only the sequences actually needed are read off disk.
#
# HOW – virtual __len__ (key design decision):
#   Naively setting __len__ = number_of_episodes causes a critical bug:
#   with only 5 seeded episodes and batch_size=50, drop_last=True produces
#   ZERO complete batches → "Average Loss: 0.0000 (over 0 steps)".
#
#   The fix used by Dreamer & PlaNet-TF: return a LARGE virtual length
#   (batch_size × train_steps) from __len__, and have __getitem__ IGNORE
#   the index — instead it draws a random valid episode each call.
#   This means:
#     • The DataLoader always has enough "items" to fill every batch.
#     • Randomness comes from __getitem__, so shuffle=False in the DataLoader.
#     • Works correctly with any number of episodes ≥ 1.
#
#   Pre-filtering still guarantees every served sequence has T >= seq_len,
#   so torch.stack in collate_episode_batch never hits a shape mismatch.
# ──────────────────────────────────────────────────────────────────────────────
class EpisodeDataset(Dataset):
    def __init__(self, storage: EpisodeStorage, seq_len: int, virtual_len: int):
        self.seq_len     = seq_len
        self.virtual_len = virtual_len

        # ── Load ALL valid episodes into RAM (once, at construction time) ────
        #
        # WHY:  Each __getitem__ call previously opened a .npz on Drive.
        #       With virtual_len=5000, that is 5000 Drive reads per iteration —
        #       at ~10 ms per read over the network that's 50 s of I/O before
        #       a single gradient step.  Loading here once costs the same
        #       per-file I/O but amortises it over the full training run.
        #
        # COST: Pendulum-v1 episode = 200 steps × 64×64×3 uint8 ≈ 2.5 MB.
        #       100 episodes ≈ 250 MB — trivial vs Colab's 12+ GB RAM.
        #
        # HOW:  We materialise each episode as a plain dict of numpy arrays
        #       so __getitem__ only does an in-memory slice + dtype cast.
        self.episodes = []   # list of {obs, action, reward, terminal}
        skipped = 0
        for path in storage.episode_paths:
            with np.load(path) as ep:
                T = len(ep['reward'])
                if T >= seq_len:
                    # Copy arrays out of the NpzFile so the file can close
                    self.episodes.append({
                        'obs'     : ep['obs'].copy(),       # (T, 64, 64, 3) uint8
                        'action'  : ep['action'].copy(),    # (T, action_dim)
                        'reward'  : ep['reward'].copy(),    # (T,)
                        'terminal': ep['terminal'].copy(),  # (T,)
                    })
                else:
                    skipped += 1

        if skipped:
            print(f"[EpisodeDataset] Skipped {skipped} short episodes "
                  f"(< {seq_len} frames).")
        if len(self.episodes) == 0:
            raise RuntimeError(
                f"No episodes with >= {seq_len} frames found. "
                f"Collect more data or reduce seq_len."
            )
        total_mb = sum(ep['obs'].nbytes for ep in self.episodes) / 1e6
        print(f"[EpisodeDataset] Cached {len(self.episodes)} episodes "
              f"({total_mb:.1f} MB) | virtual_len={virtual_len}")

    def __len__(self):
        # Virtual size: guarantees the DataLoader produces exactly
        # virtual_len // batch_size full batches per iteration.
        return self.virtual_len

    def __getitem__(self, _idx):
        # ── Recency-weighted episode selection ────────────────────────────────
        # Earlier episodes (random policy) get weight 0.5.
        # Most recent episode gets weight 1.0.
        # This biases training toward the agent's improving experience while
        # keeping early diversity — without fully discarding old episodes.
        #
        # np.linspace(0.5, 1.0, N) produces N evenly spaced values:
        #   e.g. 5 episodes → [0.5, 0.625, 0.75, 0.875, 1.0]
        # Dividing by .sum() turns them into a probability distribution that
        # sums to 1.0, usable by np.random.choice(p=...).
        n       = len(self.episodes)
        weights = np.linspace(0.5, 1.0, n)
        weights = weights / weights.sum()                              # normalise → probabilities
        idx     = np.random.choice(n, p=weights)                      # weighted draw
        ep      = self.episodes[idx]

        T     = len(ep['reward'])
        start = np.random.randint(0, T - self.seq_len + 1)
        end   = start + self.seq_len

        obs      = ep['obs'][start:end].astype(np.float32) / 255.0  # (seq_len, 64, 64, 3)
        action   = ep['action'][start:end]                           # (seq_len, action_dim)
        reward   = ep['reward'][start:end]                           # (seq_len,)
        terminal = ep['terminal'][start:end].astype(np.float32)     # (seq_len,)

        return (
            torch.from_numpy(obs),
            torch.from_numpy(action),
            torch.from_numpy(reward),
            torch.from_numpy(terminal),
        )

    def add_episode(self, path: 'pathlib.Path') -> None:
        """
        Append a newly collected episode to the in-RAM cache.

        Why this exists
        ---------------
        Without this method, train_planner must destroy and rebuild the entire
        EpisodeDataset (re-reading every .npz from Drive) at the start of each
        iteration — O(n_episodes) Drive reads per iteration.

        With add_episode(), the dataset is built ONCE before the training loop
        and only the new episode file is read and appended — O(1) Drive read
        per iteration regardless of how many total episodes exist.

        Parameters
        ----------
        path : pathlib.Path  Path to the newly written .npz file on Drive.
        """
        with np.load(path) as ep:
            T = len(ep['reward'])
            if T >= self.seq_len:
                self.episodes.append({k: np.array(ep[k]) for k in ep})
                # Also bump virtual_len so the DataLoader still produces
                # the correct number of batches next iteration.
                # (virtual_len is re-read by DataLoader at construction;
                #  we don't reconstruct the DataLoader, so this is future-proofing.)
            else:
                print(f"[Dataset] Skipped short episode ({T} < {self.seq_len} frames)")



def collate_episode_batch(batch):
    """
    WHAT: Custom collate for DataLoader.
    WHY:  Default collate stacks to (B, T, …).  WorldModel.forward expects
          (T, B, …), so we transpose here once instead of in every train step.
    HOW:  torch.stack along dim=0 then permute the time and batch dims.
    """
    obs, action, reward, terminal = zip(*batch)
    obs      = torch.stack(obs).permute(1, 0, 2, 3, 4)   # (T, B, 64, 64, 3)
    action   = torch.stack(action).permute(1, 0, 2)       # (T, B, action_dim)
    reward   = torch.stack(reward).permute(1, 0)          # (T, B)
    terminal = torch.stack(terminal).permute(1, 0)        # (T, B)
    return obs, action, reward, terminal


# ──────────────────────────────────────────────────────────────────────────────
# 10c. ORIGINAL ReplayBuffer kept for reference  [COMMENTED OUT]
#
# The buffer below lives entirely in RAM and is lost on Colab restart.
# EpisodeStorage + EpisodeDataset replaces it for the persistent workflow.
# ──────────────────────────────────────────────────────────────────────────────
# class ReplayBuffer:
#     def __init__(self, capacity, obs_shape, action_dim):
#         self.capacity      = capacity
#         self.idx           = 0
#         self.is_full       = False
#         self.obs_buffer      = np.empty((capacity, *obs_shape),   dtype=np.uint8)
#         self.action_buffer   = np.empty((capacity, action_dim),   dtype=np.float32)
#         self.reward_buffer   = np.empty((capacity,),              dtype=np.float32)
#         self.terminal_buffer = np.empty((capacity,),              dtype=bool)
#
#     def add(self, obs, action, reward, terminal):
#         self.obs_buffer[self.idx]      = obs
#         self.action_buffer[self.idx]   = action
#         self.reward_buffer[self.idx]   = reward
#         self.terminal_buffer[self.idx] = terminal
#         self.idx = (self.idx + 1) % self.capacity
#         if self.idx == 0:
#             self.is_full = True
#
#     def _is_valid(self, start, seq_len, current_capacity):
#         end = start + seq_len
#         if end > current_capacity:
#             return False
#         if self.is_full and start < self.idx < end:
#             return False
#         if self.terminal_buffer[start:end - 1].any():
#             return False
#         return True
#
#     def sample(self, batch_size, seq_len, device):
#         current_capacity = self.capacity if self.is_full else self.idx
#         if current_capacity < seq_len:
#             return None
#         indices, attempts = [], 0
#         max_attempts = batch_size * 100
#         while len(indices) < batch_size and attempts < max_attempts:
#             start = np.random.randint(0, current_capacity)
#             if self._is_valid(start, seq_len, current_capacity):
#                 indices.append(start)
#             attempts += 1
#         if len(indices) < batch_size:
#             return None
#         obs_b, act_b, rew_b, term_b = [], [], [], []
#         for start in indices:
#             end = start + seq_len
#             obs_b.append(self.obs_buffer[start:end])
#             act_b.append(self.action_buffer[start:end])
#             rew_b.append(self.reward_buffer[start:end])
#             term_b.append(self.terminal_buffer[start:end])
#         return self._post_process(
#             np.stack(obs_b), np.stack(act_b), np.stack(rew_b), np.stack(term_b), device
#         )
#
#     def _post_process(self, obs, action, reward, terminal, device):
#         obs      = torch.as_tensor(obs,      device=device).float() / 255.0
#         action   = torch.as_tensor(action,   device=device).float()
#         reward   = torch.as_tensor(reward,   device=device).float()
#         terminal = torch.as_tensor(terminal, device=device).float()
#         obs      = obs.permute(1, 0, 2, 3, 4)
#         action   = action.transpose(1, 0)
#         reward   = reward.transpose(1, 0)
#         terminal = terminal.transpose(1, 0)
#         return obs, action, reward, terminal


# ──────────────────────────────────────────────────────────────────────────────
# 11. TRAINING STEP
# ──────────────────────────────────────────────────────────────────────────────
def train_step(model, optimizer, obs, action, reward, device):
    recon_image, pred_reward, p_m, p_s, q_m, q_s, overshoot_kl = model(obs, action)

    optimizer.zero_grad()
    loss = calculate_loss(recon_image, obs, reward, pred_reward, p_m, p_s, q_m, q_s, overshoot_kl)
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=100.0)
    optimizer.step()

    return loss.item()


# ──────────────────────────────────────────────────────────────────────────────
# 12. CEM PLANNER  (Cross-Entropy Method)
# ──────────────────────────────────────────────────────────────────────────────
class CEMPlanner:
    def __init__(self, model, num_candidates=1000, top_k=100,
                 n_steps=12, iteration=10, action_dim=1):
        self.model          = model
        self.num_candidates = num_candidates
        self.top_k          = top_k
        self.n_steps        = n_steps
        self.iteration      = iteration
        self.action_dim     = action_dim
        self.epsilon        = 0.1           # minimum std to prevent collapse

    @torch.no_grad()
    def plan(self, h, s):
        if h.ndim == 1:
            h = h.unsqueeze(0)
            s = s.unsqueeze(0)
        device = h.device

        mean = torch.zeros(self.n_steps, self.action_dim, device=device)
        std  = torch.ones(self.n_steps,  self.action_dim, device=device)

        for _ in range(self.iteration):
            noise   = torch.randn(self.num_candidates, self.n_steps, self.action_dim, device=device)
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise

            # Fix 3 – Pendulum-v1 torque range is [-2, 2]
            actions = actions.clamp(-2.0, 2.0)

            h_im = h.expand(self.num_candidates, -1)
            s_im = s.expand(self.num_candidates, -1)
            total_rewards = torch.zeros(self.num_candidates, device=device)

            for t in range(self.n_steps):
                _, _, h_im, s_im = self.model.rssm.imagine_step(h_im, s_im, actions[:, t])
                reward = self.model.reward(h_im, s_im)      # (num_candidates, 1)
                total_rewards += reward.squeeze(-1)

            top_idx      = total_rewards.topk(self.top_k).indices
            elite        = actions[top_idx]                  # (top_k, n_steps, action_dim)
            mean         = elite.mean(dim=0)
            std          = elite.std(dim=0) + self.epsilon

        return mean[0]   # (action_dim,)  – best first action


# ──────────────────────────────────────────────────────────────────────────────
# 13. PIXEL WRAPPER  (wraps any Gym env to return 64×64 RGB pixel observations)
# ──────────────────────────────────────────────────────────────────────────────
class PixelWrapper(gym.Wrapper):
    def __init__(self, env, render_size=64):
        super().__init__(env)
        self.render_size = render_size
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(render_size, render_size, 3),
            dtype=np.uint8,
        )

    def _get_pixels(self):
        img = self.env.render()
        # Newer Gymnasium versions occasionally return a list of frames instead
        # of a bare numpy array.  Unpack it so .shape works reliably.
        if isinstance(img, list):
            img = img[0]
        img = np.asarray(img)           # ensure it's always a numpy array
        if img.shape[:2] != (self.render_size, self.render_size):
            img = cv2.resize(img, (self.render_size, self.render_size),
                             interpolation=cv2.INTER_AREA)
        return img

    def reset(self, **kwargs):
        _ = self.env.reset(**kwargs)
        return self._get_pixels(), {}

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._get_pixels(), reward, terminated, truncated, info


# ──────────────────────────────────────────────────────────────────────────────
# 14. CONFIG  [UPDATED – adds Google Drive paths + DataLoader settings]
# ──────────────────────────────────────────────────────────────────────────────

# Original Config kept for reference:
# class Config:
#     def __init__(self):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.seed_episodes             = 5
#         self.total_iterations          = 100
#         self.train_steps_per_iteration = 1000
#         self.collect_episodes          = 1
#         self.batch_size = 50
#         self.seq_len    = 50
#         self.capacity   = 100_000
#         self.obs_shape  = (64, 64, 3)
#         self.action_dim = 1

class Config:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── Training loop ─────────────────────────────────────────────────
        self.seed_episodes             = 5     # random episodes before training starts
        self.total_iterations          = 100   # train + collect cycles
        self.train_steps_per_iteration = 100   # gradient steps per cycle (reduced: DataLoader handles batching)
        self.collect_episodes          = 1     # new episodes to collect after each train cycle

        # ── DataLoader ────────────────────────────────────────────────────
        self.batch_size  = 50    # sequences per mini-batch
        self.seq_len     = 50    # timesteps per sequence (for BPTT)
        self.num_workers = 2     # parallel workers for Drive I/O prefetch
                                 # set to 0 on Windows / if Drive is slow

        # ── Environment ───────────────────────────────────────────────────
        self.obs_shape  = (64, 64, 3)
        self.action_dim = 1  # Fix 2 – Pendulum-v1 has a 1-D continuous action

        # ── Google Drive paths (Colab) ────────────────────────────────────
        # Change the base path to match your own Drive layout.
        # In Colab: drive.mount('/content/drive') makes this available.
        self.drive_base        = '/content/drive/MyDrive/PlaNet'
        self.episode_dir       = f'{self.drive_base}/episodes'   # .npz files go here
        self.checkpoint_dir    = f'{self.drive_base}/checkpoints' # model weights go here
        self.viz_dir           = f'{self.drive_base}/visualizations' # dream GIFs + PNGs go here
        self.checkpoint_every  = 10   # save a checkpoint every N iterations
        self.keep_checkpoints  = 5    # keep only the last N checkpoints to save Drive space


# ──────────────────────────────────────────────────────────────────────────────
# 15. DATA COLLECTION  [UPDATED – writes episodes to Drive via EpisodeStorage]
#
# WHAT:  Runs num_episodes rollouts and persists each as a .npz on Drive.
# WHY:   Under the old code, add() wrote to a RAM buffer lost on restart.
#        Now each episode is flushed to disk immediately after it finishes,
#        so partial training still has all collected data available.
# HOW:   We accumulate obs/action/reward/terminal in plain Python lists during
#        the rollout, then call storage.add_episode() once per episode to batch-
#        write the numpy arrays as a single compressed file.
#
# Original collect_experience (RAM-only) kept below for reference:
# def collect_experience(env, model, planner, replay_buffer, num_episodes, device):
#     model.eval()
#     for _ in range(num_episodes):
#         obs, _ = env.reset()
#         done   = False
#         h = torch.zeros(1, 200, device=device)
#         s = torch.zeros(1, 30,  device=device)
#         with torch.no_grad():
#             init_obs = torch.tensor(obs, dtype=torch.float32, device=device)
#             init_obs = init_obs.unsqueeze(0).unsqueeze(0) / 255.0
#             dummy_a  = torch.zeros(1, planner.action_dim, device=device)
#             _, _, _, _, h, s = model.rssm.obs_step(h, s, init_obs[0], dummy_a)
#         while not done:
#             action    = planner.plan(h, s)
#             action_np = action.cpu().numpy()
#             next_obs, reward, terminated, truncated, _ = env.step(action_np)
#             done = terminated or truncated
#             replay_buffer.add(obs, action_np, reward, done)
#             with torch.no_grad():
#                 obs_t  = torch.tensor(next_obs, dtype=torch.float32, device=device)
#                 obs_t  = obs_t.unsqueeze(0).unsqueeze(0) / 255.0
#                 act_t  = action.unsqueeze(0).unsqueeze(0)
#                 _, _, _, _, h, s = model.rssm.obs_step(h, s, obs_t[0], act_t[0])
#             obs = next_obs
# ──────────────────────────────────────────────────────────────────────────────
def collect_experience(env, model, planner, storage: EpisodeStorage, num_episodes, device):
    """Run rollouts and save each complete episode to Drive as a .npz file."""
    model.eval()

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done   = False

        # Transition accumulators for the current episode
        obs_list, action_list, reward_list, terminal_list = [], [], [], []

        h = torch.zeros(1, 200, device=device)
        s = torch.zeros(1, 30,  device=device)

        # Fix 5 – bootstrap the belief with the first real observation
        with torch.no_grad():
            init_obs = torch.tensor(obs, dtype=torch.float32, device=device)
            init_obs = init_obs.unsqueeze(0).unsqueeze(0) / 255.0
            dummy_a  = torch.zeros(1, planner.action_dim, device=device)
            _, _, _, _, h, s = model.rssm.obs_step(h, s, init_obs[0], dummy_a)

        while not done:
            action    = planner.plan(h, s)   # (action_dim,)
            action_np = action.cpu().numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            # Accumulate transition (store raw uint8 to save memory)
            obs_list.append(obs)
            action_list.append(action_np)
            reward_list.append(float(reward))
            terminal_list.append(bool(done))

            # Update belief with next observation
            with torch.no_grad():
                obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
                obs_t = obs_t.unsqueeze(0).unsqueeze(0) / 255.0
                act_t = action.unsqueeze(0).unsqueeze(0)
                _, _, _, _, h, s = model.rssm.obs_step(h, s, obs_t[0], act_t[0])

            obs = next_obs

        # ── Flush the finished episode to Google Drive ────────────────────
        storage.add_episode(obs_list, action_list, reward_list, terminal_list)


# ──────────────────────────────────────────────────────────────────────────────
# 16. EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_planer(env, model, planner, num_episode, device):
    """Run the model greedily and report episode rewards."""
    model.eval()
    total_rewards = []

    for ep in range(num_episode):
        obs, _ = env.reset()
        done   = False
        ep_reward = 0

        h = torch.zeros(1, 200, device=device)
        s = torch.zeros(1, 30,  device=device)

        # Fix 5 – encode the first real observation before planning begins
        with torch.no_grad():
            init_obs = torch.tensor(obs, dtype=torch.float32, device=device)
            init_obs = init_obs.unsqueeze(0).unsqueeze(0) / 255.0
            dummy_a  = torch.zeros(1, planner.action_dim, device=device)
            _, _, _, _, h, s = model.rssm.obs_step(h, s, init_obs[0], dummy_a)

        while not done:
            with torch.no_grad():
                action    = planner.plan(h, s)
                action_np = action.cpu().numpy()

                next_obs, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated
                ep_reward += reward

                obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
                obs_t = obs_t.unsqueeze(0).unsqueeze(0) / 255.0
                act_t = action.unsqueeze(0).unsqueeze(0)
                _, _, _, _, h, s = model.rssm.obs_step(h, s, obs_t[0], act_t[0])

        total_rewards.append(ep_reward)
        print(f"Eval Episode {ep + 1}: Reward = {ep_reward:.2f}")

    avg = np.mean(total_rewards)
    print(f"Average Evaluation Reward: {avg:.2f}")
    return avg


# ──────────────────────────────────────────────────────────────────────────────
# 17a. CHECKPOINT UTILITIES  [NEW]
#
# WHAT:  save_checkpoint() writes model weights, optimizer state, and the
#        current iteration index to Drive.  load_checkpoint() restores them
#        so training can resume after a Colab timeout.
#
# WHY:   Colab sessions disconnect after ~12 h.  Without checkpointing,
#        all gradient progress is lost.  With it, we just re-run the notebook
#        cell and pick up exactly where we left off.
#
# HOW:   torch.save stores a plain dict.  We keep the last N checkpoints
#        (config.keep_checkpoints) and delete older ones to avoid filling Drive.
# ──────────────────────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, iteration, config):
    ckpt_dir = pathlib.Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    path = ckpt_dir / f'ckpt_{iteration:06d}.pt'
    torch.save({
        'iteration' : iteration,
        'model'     : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }, path)
    print(f"  [Checkpoint] Saved → {path}")

    # ── Prune old checkpoints ────────────────────────────────────────────
    all_ckpts = sorted(ckpt_dir.glob('ckpt_*.pt'))
    for old in all_ckpts[:-config.keep_checkpoints]:  # keep the newest N
        old.unlink()
        print(f"  [Checkpoint] Deleted old → {old.name}")


def load_checkpoint(model, optimizer, config):
    """
    Loads the latest checkpoint from Drive if one exists.
    Returns the iteration number to resume from (0 if no checkpoint found).
    """
    ckpt_dir = pathlib.Path(config.checkpoint_dir)
    if not ckpt_dir.exists():
        return 0

    all_ckpts = sorted(ckpt_dir.glob('ckpt_*.pt'))
    if not all_ckpts:
        return 0

    latest = all_ckpts[-1]
    data   = torch.load(latest, map_location=config.device)
    model.load_state_dict(data['model'])
    optimizer.load_state_dict(data['optimizer'])
    start_iter = data['iteration'] + 1          # resume AFTER the saved iteration
    print(f"[Checkpoint] Resumed from {latest.name}  (iteration {start_iter})")
    return start_iter


# ──────────────────────────────────────────────────────────────────────────────
# 17b. MAIN TRAINING LOOP  [UPDATED – uses DataLoader + Drive checkpointing]
#
# WHAT:  Replaces the manual replay_buffer.sample() loop with a proper
#        PyTorch DataLoader that reads episode .npz files from Drive.
#
# WHY:   DataLoader gives us:
#          • Automatic shuffling at the episode level each epoch.
#          • Parallel prefetch (num_workers) hides Drive I/O latency.
#          • Cleaner code – no more manual index bookkeeping.
#
# HOW:   After each collect phase, we rebuild the Dataset so the DataLoader
#        sees the newly added episodes.  Checkpoints are saved every
#        config.checkpoint_every iterations so crashes never cost much work.
#
# Original train_planner (RAM buffer version) kept for reference:
# def train_planner(env, model, optimizer, planner, replay_buffer, config):
#     device = config.device
#     print("Collecting initial random seed experience...")
#     for _ in range(config.seed_episodes):
#         obs, _ = env.reset()
#         done   = False
#         while not done:
#             action = env.action_space.sample()
#             next_obs, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated
#             replay_buffer.add(obs, action, reward, done)
#             obs = next_obs
#     for iteration in range(config.total_iterations):
#         print(f"\nIteration {iteration + 1}/{config.total_iterations}")
#         model.train()
#         total_loss = 0.0
#         for _ in range(config.train_steps_per_iteration):
#             batch = replay_buffer.sample(config.batch_size, config.seq_len, device)
#             if batch is None:
#                 continue
#             obs_batch, action_batch, reward_batch, _ = batch
#             total_loss += train_step(model, optimizer, obs_batch, action_batch, reward_batch, device)
#         print(f"  Average Loss: {total_loss / config.train_steps_per_iteration:.4f}")
#         collect_experience(env, model, planner, replay_buffer, config.collect_episodes, device)
# ──────────────────────────────────────────────────────────────────────────────
def train_planner(env, model, optimizer, planner, storage: EpisodeStorage, config):
    device     = config.device
    start_iter = load_checkpoint(model, optimizer, config)  # resume if checkpoint exists

    # ── Phase 1: seed Drive with random episodes if the storage is empty ──
    if len(storage) == 0:
        print("Collecting initial random seed experience...")
        for _ in range(config.seed_episodes):
            obs, _ = env.reset()
            done   = False
            obs_l, act_l, rew_l, term_l = [], [], [], []
            while not done:
                action = env.action_space.sample()           # uniform random action
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                obs_l.append(obs)
                act_l.append(np.array(action, dtype=np.float32).reshape(-1))
                rew_l.append(float(reward))
                term_l.append(bool(done))
                obs = next_obs
            storage.add_episode(obs_l, act_l, rew_l, term_l)  # persist to Drive
        print(f"  Seeded {len(storage)} episodes.")

    # ── Phase 2: alternating train / collect ─────────────────────────────
    #
    # Persistent dataset — built ONCE here, then extended with add_episode().
    # Previously the dataset was rebuilt inside the loop from all Drive files
    # (O(n_episodes) reads per iteration).  Now only the single new episode
    # file is read per iteration — O(1) Drive I/O regardless of dataset size.
    virtual_len = config.batch_size * config.train_steps_per_iteration
    dataset     = EpisodeDataset(storage, seq_len=config.seq_len,
                                 virtual_len=virtual_len)
    loader      = DataLoader(
        dataset,
        batch_size  = config.batch_size,
        shuffle     = False,         # randomness lives inside __getitem__
        num_workers = 0,             # 0 avoids fork-deadlock in Colab
        collate_fn  = collate_episode_batch,
        pin_memory  = False,
        drop_last   = True,
    )

    for iteration in range(start_iter, config.total_iterations):
        print(f"\nIteration {iteration + 1}/{config.total_iterations}  "
              f"| Episodes in storage: {len(storage)}")

        # ── Train ────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        steps_done = 0

        for obs_b, act_b, rew_b, _ in loader:
            if steps_done >= config.train_steps_per_iteration:
                break
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)
            rew_b = rew_b.to(device)
            total_loss += train_step(model, optimizer, obs_b, act_b, rew_b, device)
            steps_done += 1

        print(f"  Average Loss: {total_loss / max(steps_done, 1):.4f}  "
              f"(over {steps_done} steps)")

        # ── Collect new episode and add it to the in-RAM dataset ──────────
        # collect_experience writes to Drive (storage) and returns the path.
        # We then call dataset.add_episode() so the new episode is immediately
        # available for the next iteration without re-reading Drive.
        collect_experience(env, model, planner, storage, config.collect_episodes, device)
        # The newest path is always the last one appended to storage:
        dataset.add_episode(storage.episode_paths[-1])

        # ── Checkpoint ───────────────────────────────────────────────────
        if (iteration + 1) % config.checkpoint_every == 0:
            save_checkpoint(model, optimizer, iteration, config)


# ──────────────────────────────────────────────────────────────────────────────
# 18. ENTRY POINT  [UPDATED – mounts Drive, uses EpisodeStorage]
#
# WHAT:  Sets up Google Drive, creates the episode storage and checkpoint
#        directories, builds all components, then runs the training loop.
#
# WHY:   Drive must be mounted *before* any path in config.drive_base is
#        accessed.  We guard with a try/except so the code still works when
#        running locally (Drive won't be available, paths default to ./local).
#
# HOW:   In Colab, just run this cell.  It will:
#          1. Mount Drive → /content/drive
#          2. Create PlaNet/episodes and PlaNet/checkpoints folders.
#          3. Load the latest checkpoint if one exists.
#          4. Collect seed episodes (skipped if episodes already on Drive).
#          5. Train for config.total_iterations, saving checkpoints every
#             config.checkpoint_every iterations.
#
# Original entry-point kept for reference:
# if __name__ == "__main__":
#     config = Config()
#     device = config.device
#     base_env = gym.make('Pendulum-v1', render_mode='rgb_array')
#     env      = PixelWrapper(base_env, render_size=64)
#     model     = WorldModel().to(device)
#     optimizer = optim.AdamW(model.parameters(), lr=1e-4)
#     planner   = CEMPlanner(model, action_dim=config.action_dim)
#     replay_buffer = ReplayBuffer(config.capacity, config.obs_shape, config.action_dim)
#     print("Starting PlaNet Training...")
#     train_planner(env, model, optimizer, planner, replay_buffer, config)
#     evaluate_planer(env, model, planner, 5, device)
# ──────────────────────────────────────────────────────────────────────────────
# 19. OPEN-LOOP DREAM VISUALIZER  [NEW]
#
# WHAT:  Implements the "Open-Loop Video Prediction" experiment from PlaNet §5.
#        The model is given a short context sequence of real frames to warm up
#        its hidden state, and then asked to "dream" many steps into the future
#        purely from its prior — without ever seeing another real observation.
#
# WHY:   This is the most direct probe of world-model quality.  A good model
#        should produce imagined futures that look physically consistent
#        (pendulum swings smoothly) even without grounding observations.
#        Comparing the dream side-by-side with real frames makes the
#        quality (or failure modes) immediately obvious.
#
# HOW:
#   Step 1 – Context phase (obs_step):
#       Feed `context_frames` real observations through the RSSM using
#       obs_step.  This uses BOTH the encoder and the GRU to form a
#       posterior belief.  The final (h, s) is a compact summary of what
#       the model *knows* from those frames.
#
#   Step 2 – Dream phase (imagine_step):
#       From that (h, s) seed, roll forward `dream_steps` using ONLY the
#       prior — no encoder, no real pixels.  We feed the actions that
#       actually occurred in the recorded episode so the imagined trajectory
#       matches the real one (making comparison fair).
#
#   Step 3 – Decode:
#       Pass every (h, s) pair — both from context and dream — through the
#       Decoder to produce predicted 64×64 RGB images.
#
#   Step 4 – Save:
#       • An animated GIF where each frame is [real | recon | dream] (rows).
#       • A static PNG contact sheet (all frames tiled) for quick inspection.
#       Both are written to config.viz_dir on Google Drive.
#
# Dependencies:
#   pip install imageio imageio-ffmpeg matplotlib
# ──────────────────────────────────────────────────────────────────────────────

def visualize_dream(
    model,
    storage,
    config,
    episode_idx:    int = 0,   # which Drive episode to visualize
    context_frames: int = 5,   # warm-up frames fed as real observations
    dream_steps:    int = 50,  # how many steps to hallucinate afterward
    save_gif:       bool = True,
    save_png:       bool = True,
):
    """
    Run open-loop dream prediction on one stored episode and save the result.

    Returns:
        gif_path (pathlib.Path | None)
        png_path (pathlib.Path | None)
    """
    import matplotlib
    matplotlib.use('Agg')               # headless backend — no display needed
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    try:
        import imageio
    except ImportError:
        raise ImportError("Run:  pip install imageio imageio-ffmpeg")

    # ── 0. Prep output folder on Drive ───────────────────────────────────────
    viz_dir = pathlib.Path(config.viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load the episode from Drive ───────────────────────────────────────
    if len(storage) == 0:
        raise RuntimeError("No episodes in storage yet — collect some data first.")
    episode_idx = episode_idx % len(storage)        # safe wrap
    ep          = np.load(storage.episode_paths[episode_idx])
    obs_ep      = ep['obs']                         # (T, 64, 64, 3) uint8
    action_ep   = ep['action']                      # (T, action_dim)  float32

    T_ep = len(obs_ep)
    total_steps = context_frames + dream_steps
    if T_ep < total_steps:
        print(f"[Viz] Episode only has {T_ep} frames; "
              f"trimming total to {T_ep} (context={context_frames}, "
              f"dream={min(dream_steps, T_ep - context_frames)}).")
        dream_steps = max(0, T_ep - context_frames)

    device = config.device
    model.eval()

    # Storage for decoded frames
    real_frames   = []   # raw uint8 pixels from the episode
    recon_frames  = []   # decoder output during context phase
    dream_frames  = []   # decoder output during dream phase

    # ── 2. Context phase — warm up hidden state using real observations ───────
    h = torch.zeros(1, 200, device=device)
    s = torch.zeros(1, 30,  device=device)

    with torch.no_grad():
        for t in range(context_frames):
            # Real observation → float tensor (1, 64, 64, 3)
            obs_t = torch.tensor(obs_ep[t], dtype=torch.float32, device=device)
            obs_t = obs_t.unsqueeze(0) / 255.0

            # Previous action (a_{t-1}); zeros for t=0
            prev_a = (torch.tensor(action_ep[t - 1],
                                   dtype=torch.float32, device=device).unsqueeze(0)
                      if t > 0
                      else torch.zeros(1, config.action_dim, device=device))

            # RSSM observation step — uses encoder + GRU + posterior
            _, _, _, _, h, s = model.rssm.obs_step(h, s, obs_t, prev_a)

            # Decode posterio latent to pixel space
            recon_t = model.decoder(h, s)           # (1, 64, 64, 3) in [0,1]
            recon_t = (recon_t[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            real_frames.append(obs_ep[t])            # keep original uint8
            recon_frames.append(recon_t)

    # ── 3. Dream phase — imagine from the prior; no real observations ─────────
    with torch.no_grad():
        for d in range(dream_steps):
            t = context_frames + d                   # global timestep in episode

            # Use the action that actually happened (keeps comparison fair)
            if t < len(action_ep):
                a_t = torch.tensor(action_ep[t - 1],
                                   dtype=torch.float32, device=device).unsqueeze(0)
            else:
                a_t = torch.zeros(1, config.action_dim, device=device)

            # RSSM imagination step — ONLY the prior, no encoder
            _, _, h, s = model.rssm.imagine_step(h, s, a_t)

            # Decode the imagined latent to pixels
            dream_t = model.decoder(h, s)           # (1, 64, 64, 3) in [0,1]
            dream_t = (dream_t[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            dream_frames.append(dream_t)
            # Real frame for comparison (if it exists in the episode)
            if t < len(obs_ep):
                real_frames.append(obs_ep[t])
            else:
                real_frames.append(np.zeros_like(real_frames[0]))  # black placeholder

    # ── 4. Render and save ────────────────────────────────────────────────────
    # Pad recon with blank frames to match real_frames length during dream phase
    recon_placeholder = np.zeros_like(recon_frames[0])
    recon_all = recon_frames + [recon_placeholder] * dream_steps

    blank64  = np.zeros((64, 64, 3), dtype=np.uint8)
    gif_path = None
    png_path = None

    # ─── 4a. GIF — 256 px panels, labelled phase banners ─────────────────────
    if save_gif:
        SCALE   = 256    # px per panel
        SEP_W   = 10     # separator px
        BANNER  = 22     # banner height (holds readable text)

        sep        = np.ones((SCALE, SEP_W, 3), dtype=np.uint8) * 200
        col_titles = ['REAL (ground truth)', 'RECON (posterior)', 'DREAM (prior)']
        gif_frames = []

        for i in range(len(real_frames)):
            is_dream     = (i >= context_frames)
            phase_colour = (0, 160, 0) if not is_dream else (180, 0, 0)
            phase_text   = f'CONTEXT  t={i}' if not is_dream else f'DREAM  t={i}'

            panels = [
                cv2.resize(real_frames[i],                                            (SCALE, SCALE), interpolation=cv2.INTER_NEAREST),
                cv2.resize(recon_all[i],                                              (SCALE, SCALE), interpolation=cv2.INTER_NEAREST),
                cv2.resize(dream_frames[i - context_frames] if is_dream else blank64, (SCALE, SCALE), interpolation=cv2.INTER_NEAREST),
            ]

            for p_idx, panel in enumerate(panels):
                panel[:BANNER, :] = phase_colour
                cv2.putText(panel, col_titles[p_idx],
                            (4, BANNER - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (255, 255, 255), 1, cv2.LINE_AA)

            # Phase + timestep label on the Real panel below the banner
            cv2.putText(panels[0], phase_text,
                        (4, BANNER + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.46, (255, 255, 0), 1, cv2.LINE_AA)

            gif_frames.append(np.hstack([panels[0], sep, panels[1], sep, panels[2]]))

        gif_path = viz_dir / f'dream_ep{episode_idx:04d}.gif'
        imageio.mimsave(str(gif_path), gif_frames, fps=5, loop=0)
        print(f"[Viz] GIF saved → {gif_path}  ({SCALE*3 + SEP_W*2}×{SCALE}px, {len(gif_frames)} frames)")

    # ─── 4b. PNG — 16 curated large frames, dark background, 180 dpi ─────────
    # We pick 5 context + 5 early dream + gap + 5 late dream = 16 columns.
    # Each cell is 3 inches so every image is big and legible.
    if save_png:
        N_SHOW    = 5      # how many early / late dream frames to show
        ctx_idxs  = list(range(context_frames))
        early_idxs= list(range(context_frames, context_frames + min(N_SHOW, dream_steps)))
        late_start = context_frames + dream_steps - min(N_SHOW, dream_steps)
        late_idxs = list(range(late_start, context_frames + dream_steps))
        has_gap   = (context_frames + N_SHOW) < late_start
        all_idxs  = ctx_idxs + early_idxs + (['gap'] if has_gap else []) + late_idxs
        n_cols    = len(all_idxs)

        CELL = 3.0     # inches per image — very large
        LBLW = 1.6     # reserved for row labels on the left

        fig = plt.figure(figsize=(LBLW + n_cols * CELL, 3 * CELL + 1.8),
                         facecolor='#181818')
        gs  = fig.add_gridspec(3, n_cols,
                               left  = LBLW / (LBLW + n_cols * CELL),
                               right = 0.99, top = 0.87, bottom = 0.06,
                               wspace=0.07, hspace=0.22)

        ROW_META = [
            ('Real\nGround Truth',           '#43a047'),
            ('Recon\nPosterior q(s|h,e)',     '#1e88e5'),
            ('Dream\nPrior p(s|h) only',      '#e53935'),
        ]

        for ri, (row_lbl, row_col) in enumerate(ROW_META):
            for ci, t_idx in enumerate(all_idxs):
                ax = fig.add_subplot(gs[ri, ci])
                ax.set_facecolor('#181818')
                ax.set_xticks([]); ax.set_yticks([])

                if t_idx == 'gap':
                    ax.axis('off')
                    ax.text(0.5, 0.5, '···\ntime\nskipped', ha='center', va='center',
                            fontsize=13, color='#888888', transform=ax.transAxes,
                            linespacing=1.6, fontstyle='italic')
                    if ri == 0:
                        ax.set_title('···', fontsize=12, color='#666666', pad=7)
                    continue

                t        = t_idx
                is_ctx   = (t < context_frames)
                bdr_col  = '#66bb6a' if is_ctx else '#ef5350'

                if   ri == 0: img = real_frames[t]
                elif ri == 1: img = recon_all[t]    if is_ctx   else blank64
                else:         img = dream_frames[t - context_frames] if not is_ctx else blank64

                ax.imshow(img, interpolation='nearest', aspect='equal')

                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_edgecolor(bdr_col); sp.set_linewidth(3.5)

                if ri == 0:   # column header on top row only
                    tag = 'CTX' if is_ctx else 'DREAM'
                    ax.set_title(f't = {t}\n[ {tag} ]', fontsize=12, pad=6,
                                 color=bdr_col, fontweight='bold')

            # Row label in left margin
            y_frac = 1.0 - (ri + 0.5) / 3.0
            fig.text(0.005, 0.06 + y_frac * 0.81, row_lbl,
                     ha='left', va='center', fontsize=12, color=row_col,
                     fontweight='bold', rotation=90, rotation_mode='anchor')

        fig.suptitle(
            f'Open-Loop Dream  ·  Episode {episode_idx}  ·  '
            f'{context_frames} context  +  {dream_steps} dream steps\n'
            f'Columns: all {len(ctx_idxs)} context  |  first {len(early_idxs)} dream  |  '
            + ('···  |  ' if has_gap else '') +
            f'last {len(late_idxs)} dream',
            fontsize=14, color='white', y=0.97,
        )

        from matplotlib.patches import Patch
        fig.legend(handles=[
            Patch(facecolor='#66bb6a', label='Context — real observations fed in'),
            Patch(facecolor='#ef5350', label='Dream   — prior only, no real pixels'),
        ], loc='lower center', ncol=2, fontsize=11,
           facecolor='#2a2a2a', edgecolor='#555', labelcolor='white',
           bbox_to_anchor=(0.5, 0.0))

        png_path = viz_dir / f'dream_grid_ep{episode_idx:04d}.png'
        fig.savefig(str(png_path), dpi=180, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[Viz] PNG grid saved → {png_path}")

    return gif_path, png_path


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Step 0: Mount Google Drive (Colab only) ───────────────────────────
    try:
        from google.colab import drive  # only available inside Colab
        drive.mount('/content/drive', force_remount=False)
        print("[Drive] Mounted at /content/drive")
    except ImportError:
        # Running locally – redirect Drive paths to a local ./local_storage folder
        print("[Drive] Not in Colab – using local fallback paths.")
        Config.drive_base     = './local_storage'

    config = Config()
    device = config.device
    print(f"[Config] Device: {device} | Episode dir: {config.episode_dir}")

    # ── Step 1: Build environment (uses gymnasium, not old gym) ──────────
    base_env = gym.make('Pendulum-v1', render_mode='rgb_array')
    env      = PixelWrapper(base_env, render_size=64)

    # ── Step 2: Build model components ───────────────────────────────────
    model     = WorldModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    planner   = CEMPlanner(model, action_dim=config.action_dim)

    # ── Step 3: Create Drive-backed episode storage ───────────────────────
    # EpisodeStorage scans the Drive folder on init and picks up existing .npz
    # files, so re-running the notebook after a restart just continues adding.
    storage = EpisodeStorage(config.episode_dir)

    # ── Step 4: Train ─────────────────────────────────────────────────────
    print("Starting PlaNet Training...")
    train_planner(env, model, optimizer, planner, storage, config)

    # ── Step 5: Evaluate and save final weights ───────────────────────────
    evaluate_planer(env, model, planner, 5, device)
    save_checkpoint(model, optimizer, config.total_iterations - 1, config)  # final save

    # ── Step 6: Open-loop dream visualisation ────────────────────────────
    # Runs on the first 3 stored episodes so you can compare different
    # starting conditions.  Saves GIF + PNG to config.viz_dir on Drive.
    print("\nGenerating dream visualisations...")
    for ep_idx in range(min(3, len(storage))):
        visualize_dream(
            model,
            storage,
            config,
            episode_idx    = ep_idx,
            context_frames = 5,    # warm up for 5 real frames
            dream_steps    = 50,   # then imagine 50 steps into the future
            save_gif       = True,
            save_png       = True,
        )
    print("Done.  Check your Drive at:", config.viz_dir)
