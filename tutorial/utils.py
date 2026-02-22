# =============================================================================
#  utils.py  —  PlaNet utilities: environment wrapper, storage, checkpoints,
#               and the dream visualizer.  All model code lives in the notebook.
# =============================================================================

import pathlib
import numpy as np
import cv2
import torch
import gymnasium as gym
from gymnasium import spaces


# ── PIXEL WRAPPER ─────────────────────────────────────────────────────────────
class PixelWrapper(gym.Wrapper):
    """Makes any Gymnasium env return 64×64 RGB pixel observations."""
    def __init__(self, env, render_size=64):
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
        return img

    def reset(self, **kwargs):
        _ = self.env.reset(**kwargs)
        return self._get_pixels(), {}

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._get_pixels(), reward, terminated, truncated, info


# ── EPISODE STORAGE ───────────────────────────────────────────────────────────
class EpisodeStorage:
    """Saves each episode as a compressed .npz on disk / Google Drive."""
    def __init__(self, episode_dir: str):
        self.episode_dir   = pathlib.Path(episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.episode_paths = sorted(self.episode_dir.glob('ep_*.npz'))
        print(f"[Storage] Found {len(self.episode_paths)} episodes in {episode_dir}")

    def __len__(self):
        return len(self.episode_paths)

    def add_episode(self, obs, actions, rewards, terminals):
        path = self.episode_dir / f'ep_{len(self.episode_paths):05d}.npz'
        np.savez_compressed(path,
            obs      = np.array(obs,       dtype=np.uint8),
            action   = np.array(actions,   dtype=np.float32),
            reward   = np.array(rewards,   dtype=np.float32),
            terminal = np.array(terminals, dtype=bool))
        self.episode_paths.append(path)


# ── CHECKPOINTING ─────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, iteration, config):
    ckpt_dir = pathlib.Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f'ckpt_{iteration:06d}.pt'
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration}, path)
    print(f"  [Checkpoint] Saved → {path.name}")
    for old in sorted(ckpt_dir.glob('ckpt_*.pt'))[:-config.keep_checkpoints]:
        old.unlink()


def load_checkpoint(model, optimizer, config):
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


# ── DREAM VISUALIZER ──────────────────────────────────────────────────────────
def visualize_dream(model, storage, config,
                    episode_idx=0, context_frames=5, dream_steps=50,
                    save_gif=True, save_png=True):
    """
    Open-loop prediction:
      Context phase — feed real frames → posterior state (green)
      Dream   phase — prior only, no encoder                (red)
    Saves GIF + PNG contact sheet to config.viz_dir.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        import imageio
    except ImportError:
        raise ImportError("pip install imageio imageio-ffmpeg")

    viz_dir = pathlib.Path(config.viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    episode_idx = episode_idx % len(storage)
    ep          = np.load(storage.episode_paths[episode_idx])
    obs_ep      = ep['obs']
    action_ep   = ep['action']
    dream_steps = max(0, min(dream_steps, len(obs_ep) - context_frames))
    device      = config.device
    model.eval()

    real_frames, recon_frames, dream_frames_list = [], [], []
    blank = np.zeros((64, 64, 3), dtype=np.uint8)

    h = torch.zeros(1, 200, device=device)
    s = torch.zeros(1, 30,  device=device)

    with torch.no_grad():
        for t in range(context_frames):
            obs_t  = torch.tensor(obs_ep[t], dtype=torch.float32,
                                  device=device).unsqueeze(0) / 255.0
            prev_a = (torch.tensor(action_ep[t-1], dtype=torch.float32,
                                   device=device).unsqueeze(0)
                      if t > 0 else torch.zeros(1, config.action_dim, device=device))
            _, _, _, _, h, s = model.rssm.obs_step(h, s, obs_t, prev_a)
            recon = (model.decoder(h, s)[0].cpu().numpy()*255).clip(0,255).astype(np.uint8)
            real_frames.append(obs_ep[t]); recon_frames.append(recon)

        for d in range(dream_steps):
            t   = context_frames + d
            a_t = (torch.tensor(action_ep[t-1], dtype=torch.float32,
                                device=device).unsqueeze(0)
                   if t < len(action_ep) else torch.zeros(1, config.action_dim, device=device))
            _, _, h, s = model.rssm.imagine_step(h, s, a_t)
            dream = (model.decoder(h, s)[0].cpu().numpy()*255).clip(0,255).astype(np.uint8)
            dream_frames_list.append(dream)
            real_frames.append(obs_ep[t] if t < len(obs_ep) else blank)

    recon_all = recon_frames + [blank] * dream_steps
    gif_path = png_path = None

    # GIF — 256px panels with phase banners
    if save_gif:
        SZ, SW, BH = 256, 10, 22
        sep  = np.ones((SZ, SW, 3), np.uint8) * 200
        cols = ['REAL (ground truth)', 'RECON (posterior)', 'DREAM (prior)']
        frames = []
        for i in range(len(real_frames)):
            is_d   = i >= context_frames
            colour = (0,160,0) if not is_d else (180,0,0)
            panels = [
                cv2.resize(real_frames[i],                                      (SZ,SZ), interpolation=cv2.INTER_NEAREST),
                cv2.resize(recon_all[i],                                        (SZ,SZ), interpolation=cv2.INTER_NEAREST),
                cv2.resize(dream_frames_list[i-context_frames] if is_d else blank, (SZ,SZ), interpolation=cv2.INTER_NEAREST),
            ]
            for pi, p in enumerate(panels):
                p[:BH] = colour
                cv2.putText(p, cols[pi], (4, BH-5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(panels[0], f'{"CONTEXT" if not is_d else "DREAM"}  t={i}',
                        (4, BH+18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,0), 1, cv2.LINE_AA)
            frames.append(np.hstack([panels[0], sep, panels[1], sep, panels[2]]))
        gif_path = viz_dir / f'dream_ep{episode_idx:04d}.gif'
        imageio.mimsave(str(gif_path), frames, fps=5, loop=0)
        print(f"[Viz] GIF → {gif_path}")

    # PNG — curated large contact sheet
    if save_png:
        N = 5
        ctx   = list(range(context_frames))
        early = list(range(context_frames, context_frames + min(N, dream_steps)))
        ls    = context_frames + dream_steps - min(N, dream_steps)
        late  = list(range(ls, context_frames + dream_steps))
        gap   = (context_frames + N) < ls
        cols  = ctx + early + (['gap'] if gap else []) + late
        nc    = len(cols)

        fig = plt.figure(figsize=(1.6 + nc*3, 3*3+1.8), facecolor='#181818')
        gs  = fig.add_gridspec(3, nc, left=1.6/(1.6+nc*3), right=0.99,
                               top=0.87, bottom=0.06, wspace=0.07, hspace=0.22)
        RM  = [('Real\nGround Truth','#43a047'),
               ('Recon\nPosterior q(s|h,e)','#1e88e5'),
               ('Dream\nPrior p(s|h) only','#e53935')]

        for ri, (rl, rc) in enumerate(RM):
            for ci, ti in enumerate(cols):
                ax = fig.add_subplot(gs[ri, ci])
                ax.set_facecolor('#181818'); ax.set_xticks([]); ax.set_yticks([])
                if ti == 'gap':
                    ax.axis('off')
                    ax.text(0.5,0.5,'···\nskipped',ha='center',va='center',
                            fontsize=13,color='#888',transform=ax.transAxes,fontstyle='italic')
                    if ri==0: ax.set_title('···',fontsize=12,color='#666',pad=7)
                    continue
                ic  = ti < context_frames
                bc  = '#66bb6a' if ic else '#ef5350'
                if   ri==0: img = real_frames[ti]
                elif ri==1: img = recon_all[ti]   if ic else blank
                else:       img = dream_frames_list[ti-context_frames] if not ic else blank
                ax.imshow(img, interpolation='nearest', aspect='equal')
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_edgecolor(bc); sp.set_linewidth(3.5)
                if ri==0:
                    ax.set_title(f't={ti}\n[{"CTX" if ic else "DREAM"}]',
                                 fontsize=12,pad=6,color=bc,fontweight='bold')
            yf = 1.0-(ri+0.5)/3.0
            fig.text(0.005, 0.06+yf*0.81, rl, ha='left', va='center',
                     fontsize=12, color=rc, fontweight='bold',
                     rotation=90, rotation_mode='anchor')

        fig.suptitle(f'Open-Loop Dream  ·  Episode {episode_idx}  ·'
                     f'  {context_frames} context + {dream_steps} dream steps',
                     fontsize=14, color='white', y=0.97)
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(facecolor='#66bb6a',label='Context — real obs'),
                             Patch(facecolor='#ef5350',label='Dream — prior only')],
                   loc='lower center', ncol=2, fontsize=11,
                   facecolor='#2a2a2a', edgecolor='#555', labelcolor='white',
                   bbox_to_anchor=(0.5,0.0))
        png_path = viz_dir / f'dream_grid_ep{episode_idx:04d}.png'
        fig.savefig(str(png_path), dpi=180, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[Viz] PNG → {png_path}")

    return gif_path, png_path
