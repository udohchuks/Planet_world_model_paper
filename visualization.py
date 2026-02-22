import pathlib
import numpy as np
import torch
import cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import os

# ──────────────────────────────────────────────────────────────────────────────

#  VISUALIZE DREAM TO END OF EPISODE
# ──────────────────────────────────────────────────────────────────────────────
def visualize_dream_to_end(model, storage, config, episode_idx=0, context_frames=5):
    """
    Open-loop prediction visualizer that runs until the end of the episode.
    Context (green) — real obs fed through encoder → posterior warm-up.
    Dream   (red)   — prior only, NO encoder, NO real pixels.
    Runs for all remaining steps in the TRUE recorded episode.
    """
    viz_dir = pathlib.Path(config.viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    episode_idx = episode_idx % len(storage)
    ep          = np.load(storage.episode_paths[episode_idx])
    obs_ep      = ep['obs']
    action_ep   = ep['action']
    
    # We dream for exactly how many steps are left in the recorded episode
    dream_steps = max(0, len(obs_ep) - context_frames)
    
    device      = config.device
    model.eval()

    real_frames, recon_frames, dream_frames_list = [], [], []
    blank = np.zeros((64, 64, 3), dtype=np.uint8)

    # Note: MuJoCo models use cfg.hidden_size, pendulums use fixed 200
    h_size = getattr(config, 'hidden_size', 200)
    s_size = getattr(config, 'state_size', 30)
    a_dim  = getattr(config, 'action_dim', 1)

    h = torch.zeros(1, h_size, device=device)
    s = torch.zeros(1, s_size, device=device)

    with torch.no_grad():
        for t in range(context_frames):
            obs_t  = torch.tensor(obs_ep[t], dtype=torch.float32, device=device).unsqueeze(0) / 255.0
            prev_a = torch.tensor(action_ep[t-1], dtype=torch.float32, device=device).unsqueeze(0) if t > 0 else torch.zeros(1, a_dim, device=device)
            _, _, _, _, h, s = model.rssm.obs_step(h, s, obs_t, prev_a)
            recon = (model.decoder(h, s)[0].cpu().numpy()*255).clip(0,255).astype(np.uint8)
            real_frames.append(obs_ep[t])
            recon_frames.append(recon)

        for d in range(dream_steps):
            t   = context_frames + d
            # We use the TRUE action recorded from the episode to feed into the dream
            a_t = torch.tensor(action_ep[t-1], dtype=torch.float32, device=device).unsqueeze(0) if t < len(action_ep) else torch.zeros(1, a_dim, device=device)
            _, _, h, s = model.rssm.imagine_step(h, s, a_t)
            dream = (model.decoder(h, s)[0].cpu().numpy()*255).clip(0,255).astype(np.uint8)
            dream_frames_list.append(dream)
            real_frames.append(obs_ep[t] if t < len(obs_ep) else blank)

    recon_all = recon_frames + [blank] * dream_steps
    
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
        
    gif_path = viz_dir / f'dream_full_ep{episode_idx:04d}.gif'
    imageio.mimsave(str(gif_path), frames, fps=15, loop=0)
    print(f"[Viz] Full Dream GIF → {gif_path}")
    return gif_path


# ──────────────────────────────────────────────────────────────────────────────
#  VISUALIZE REAL AGENT ROLLOUT USING TRAINED PLANNER
# ──────────────────────────────────────────────────────────────────────────────
def visualize_real_agent_rollout(env, model, planner, config, max_steps=1000):
    """
    Visualizes the real agent interacting with the environment using the trained model's planner.
    Note: This runs a LIVE simulation in the environment, rather than reading from storage.
    """
    viz_dir = pathlib.Path(config.viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    device = config.device
    model.eval()
    
    obs, _ = env.reset()
    done = False
    
    h_size = getattr(config, 'hidden_size', 200)
    s_size = getattr(config, 'state_size', 30)
    a_dim  = getattr(config, 'action_dim', 1)
    
    h = torch.zeros(1, h_size, device=device)
    s = torch.zeros(1, s_size,  device=device)
    
    frames = []
    
    # 1. Warm up the first step
    with torch.no_grad():
        init_obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) / 255.0
        dummy_a  = torch.zeros(1, a_dim, device=device)
        _, _, _, _, h, s = model.rssm.obs_step(h, s, init_obs[0], dummy_a)
        frames.append(obs)
        
    steps = 0
    while not done and steps < max_steps:
        with torch.no_grad():
            # Support both architectures
            action = planner.plan(h, s) if not hasattr(planner, 'cfg') else planner.plan(h, s, device)
            action_np = action.cpu().numpy().reshape(-1)
            
            # Step the real environment
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            frames.append(next_obs)
            
            # Feed the real frame back into the model to prepare for the *next* plan
            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) / 255.0
            act_t = action.unsqueeze(0).unsqueeze(0)
            _, _, _, _, h, s = model.rssm.obs_step(h, s, obs_t[0], act_t[0])
            
            steps += 1
            
    SZ, SW, BH = 256, 10, 22
    gif_frames = []
    
    for i, frame in enumerate(frames):
        panel = cv2.resize(frame, (SZ, SZ), interpolation=cv2.INTER_NEAREST)
        panel[:BH] = (0, 160, 0)
        
        cv2.putText(panel, 'AGENT LIVE ROLLOUT', (4, BH-5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(panel, f'Step: {i}', (4, BH+18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,0), 1, cv2.LINE_AA)
        
        if done and i == len(frames) - 1:
            panel[:BH] = (0, 0, 160)
            cv2.putText(panel, 'TERMINATED', (4, BH-5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
            
        gif_frames.append(panel)
        
    gif_path = viz_dir / f'agent_live_rollout.gif'
    
    # We use a higher FPS (usually 30 or 50) for a smoother playback since it's a real rollout
    imageio.mimsave(str(gif_path), gif_frames, fps=30, loop=0)
    print(f"[Viz] Agent Rollout GIF ({len(frames)} frames) → {gif_path}")
    
    return gif_path

# ──────────────────────────────────────────────────────────────────────────────
#  RUN VISUALIZATIONS WITHOUT TRAINING
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        from pyvirtualdisplay import Display
        Display(visible=0, size=(1400, 900)).start()
        print('[Display] Virtual display started')
    except ImportError:
        pass

    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
    except ImportError:
        pass

    from planet import Config, WorldModel, EpisodeStorage, CEMPlanner, PixelWrapper, load_checkpoint
    import gymnasium as gym
    import torch.optim as optim
    
    config = Config()
    device = config.device
    print(f"\n[Viz] Starting standalone visualization on {device}...")
    
    # 1. Environment (matches planet.py)
    base_env = gym.make('Pendulum-v1', render_mode='rgb_array')
    env      = PixelWrapper(base_env, render_size=64)
    
    # 2. Model & Storage
    model   = WorldModel().to(device)
    planner = CEMPlanner(model, action_dim=config.action_dim)
    storage = EpisodeStorage(config.episode_dir)
    
    # 3. Load Checkpoint
    dummy_optimizer = optim.AdamW(model.parameters(), lr=1e-4) # required for signature
    start_iter = load_checkpoint(model, dummy_optimizer, config)
    print(f"[Viz] Loaded model weights from iteration: {start_iter}")
    
    print("\n--- Running Open-Loop Dream to End ---")
    if len(storage) > 0:
        visualize_dream_to_end(model, storage, config, episode_idx=len(storage)-1, context_frames=5)
    else:
        print("[Viz] Empty storage - skipping dream visualization.")
        
    print("\n--- Running Live Agent Rollout ---")
    visualize_real_agent_rollout(env, model, planner, config, max_steps=150)
    
    print("\n[Viz] All Done! Check your Drive.")
