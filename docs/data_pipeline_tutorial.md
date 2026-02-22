# Dataset & DataLoader Deep Dive
## How PlaNet's data pipeline works — and why your RAM is fine

---

## The Big Picture

```
Environment
    │  step() → (obs, action, reward, done)
    ▼
EpisodeStorage          ← writes one .npz per episode to Google Drive
    │  episode_000001.npz  (obs, action, reward, terminal)
    │  episode_000002.npz
    │  ...
    ▼
EpisodeDataset          ← loads all .npz files into RAM ONCE per iteration
    │  random subsequence of length seq_len
    ▼
DataLoader              ← batches subsequences, permutes (B,T)→(T,B)
    ▼
WorldModel.forward()    ← processes the (T, B, ...) tensors
```

---

## 1. Why Google Drive? (It's not about RAM)

This is the key misconception. **Drive is not used to save RAM — it's used to survive Colab restarts.**

Colab gives you a **free but ephemeral VM**. Every time the runtime disconnects (timeout, crash, manual restart), everything in `/content/` is wiped. That includes any episodes you collected and any model weights you trained.

Without Drive persistence, every restart means starting from scratch:
- Re-collect 5 seed episodes
- Re-train 100 iterations
- Lose all progress

With Drive:
```
Session 1: collect 5 seed episodes, train 20 iterations → save checkpoint
         ↓ Colab disconnects at 3am
Session 2: load_checkpoint() finds ckpt_000019.pt → resume at iteration 20
           EpisodeStorage finds 25 existing .npz files → skip re-collection
         ↓ continue training where you left off
```

**Drive = free persistent hard disk, not a RAM-saving device.**

---

## 2. EpisodeStorage — Writing to Drive

```python
class EpisodeStorage:
    def add_episode(self, obs_list, action_list, reward_list, terminal_list):
        path = self.episode_dir / f'episode_{idx:06d}.npz'
        np.savez_compressed(path,
            obs      = np.array(obs_list,    dtype=np.uint8),    # uint8 = 1 byte/pixel
            action   = np.array(action_list, dtype=np.float32),
            reward   = np.array(reward_list, dtype=np.float32),
            terminal = np.array(terminal_list, dtype=bool),
        )
```

### Why `.npz` + `uint8`?

One Pendulum episode = 200 steps × 64×64×3 pixels:

| dtype   | size per episode | 100 episodes |
|---------|-----------------|--------------|
| float32 | ~9.8 MB         | ~980 MB      |
| **uint8**   | **~2.5 MB**         | **~250 MB**  |

Storing as `uint8 [0, 255]` instead of `float32 [0.0, 1.0]` saves **4×** the disk space on Drive. The conversion back to float happens in `__getitem__` in one line:

```python
obs = ep['obs'][i:i+seq_len].astype(np.float32) / 255.0
```

`savez_compressed` adds zlib compression on top, typically shrinking `.npz` files by another 30–50%.

---

## 3. EpisodeDataset — The RAM Cache

At the start of each training iteration, `EpisodeDataset.__init__` loads **all** stored episodes from Drive into RAM:

```python
for path in storage.episode_paths:
    with np.load(path) as ep:
        if len(ep['reward']) >= seq_len:
            self.episodes.append({k: np.array(ep[k]) for k in ep})
```

### Why load everything into RAM?

Without caching, every `__getitem__` call would open a `.npz` file on Drive:

```
DataLoader calls __getitem__ → open Drive file → read it → close it
                             → repeat for every sample in every batch
```

With `batch_size=16`, `train_steps=50`, `virtual_len=800`:
- **800 Drive reads per iteration**
- Drive read latency over Colab network mount: ~10–50 ms per file
- Total I/O time: **8–40 seconds** of doing nothing before a single gradient step

With RAM caching:
- **1 Drive read per episode**, once at iteration start
- All 800 `__getitem__` calls are pure in-memory array slices
- Each call takes ~1 μs instead of ~10 ms

### Why isn't your RAM exploding?

Let's count. After 100 iterations of Pendulum training:

```
Episodes stored ≈ 5 seed + 100 CEM = ~105 episodes
Each episode    ≈ 200 steps × 64×64×3 × 1 byte (uint8) = 2.46 MB
Total in RAM    ≈ 105 × 2.46 MB ≈ 258 MB
```

**258 MB out of Colab's 12 GB** — that's ~2%. Not even close to exploding.

For MuJoCo (HalfCheetah, 1000 steps per episode):
```
1000 steps × 64×64×3 = 12.3 MB per episode
After 200 episodes    = ~2.4 GB
```

Still within Colab's 12 GB limit, though worth keeping an eye on for very long runs.

---

## 4. The Virtual Length Trick

This is the most non-obvious part of the design.

### The problem

`DataLoader` needs `__len__` to know how many batches to create. If you return the actual number of episodes:

```python
def __len__(self):
    return len(self.episodes)   # e.g. 5 after seeding
```

With `batch_size=16` and `drop_last=True`, the DataLoader needs at least 16 items to form one complete batch. With only 5 episodes, it produces **zero batches** — and you get:

```
Average Loss: 0.0000 (over 0 steps)  ← silent bug!
```

### The fix: virtual length

```python
def __len__(self):
    return self.virtual_len   # = batch_size × train_steps = 800

def __getitem__(self, _idx):
    # Ignore _idx entirely — pick a random episode from RAM
    ep = self.episodes[np.random.randint(len(self.episodes))]
    ...
```

By reporting `virtual_len = batch_size × train_steps_per_iter`, the DataLoader always produces exactly `train_steps_per_iter` complete batches regardless of how many real episodes exist. The randomness is handled inside `__getitem__`, not by the DataLoader's sampler — so `shuffle=False` in the DataLoader constructor.

```
DataLoader asks: "how many items exist?" → 800
DataLoader asks: "give me item 0"  → random episode slice
DataLoader asks: "give me item 1"  → different random slice
...
DataLoader asks: "give me item 799" → another random slice
→ 800 / 16 = 50 complete batches ✓  (even with only 5 real episodes)
```

---

## 5. collate_fn — The Shape Permutation

PyTorch's default `collate_fn` stacks items from a batch into shape `(B, T, ...)`.
`WorldModel.forward()` loops over the **time axis first**, expecting `(T, B, ...)`.

Without the permutation:
```
Default collate: obs shape = (B=16, T=25, 64, 64, 3)
WorldModel sees: obs_seq[t] → shape (25, 64, 64, 3) ← wrong! t indexes into batch
```

With `collate_episode_batch`:
```python
def collate_episode_batch(batch):
    obs, action, reward, terminal = zip(*batch)
    return (
        torch.stack(obs).permute(1, 0, 2, 3, 4),  # (B,T,64,64,3) → (T,B,64,64,3)
        torch.stack(action).permute(1, 0, 2),       # (B,T,A) → (T,B,A)
        torch.stack(reward).permute(1, 0),           # (B,T)   → (T,B)
        torch.stack(terminal).permute(1, 0),         # (B,T)   → (T,B)
    )
```

Now `obs_seq[t]` correctly gives the batch of observations at timestep `t`:
```
obs_seq[t] → shape (B=16, 64, 64, 3) ← correct: 16 obs from the same timestep
```

---

## 6. The Online Learning Loop

PlaNet's data pipeline is **online** — the dataset grows while you train:

```
Iteration 1:  load  5 episodes → train → collect 1 new episode → 6 total on Drive
Iteration 2:  load  6 episodes → train → collect 1 new episode → 7 total
...
Iteration 100: load 104 episodes → train → collect 1 → 105 total
```

This is why `EpisodeDataset` is **reconstructed at the start of every iteration** inside `train_planner`:

```python
for iteration in range(start_iter, config.total_iterations):
    dataset = EpisodeDataset(storage, seq_len, virtual_len)  # re-read Drive
    loader  = DataLoader(dataset, ...)
    ...
    collect_experience(...)   # adds 1 new .npz to Drive
```

If the dataset were built once before the loop, newly collected episodes would never enter training. Rebuilding it each iteration ensures the model trains on its own improving data.

---

## 7. Complete Data Flow — Annotated

```python
# ── Training iteration ──────────────────────────────────────────────────────

# Step A: Load all Drive episodes into RAM (takes ~0.1s for 100 episodes)
dataset = EpisodeDataset(storage, seq_len=25, virtual_len=800)
# dataset.episodes = [{obs:(200,64,64,3), action:(200,1), ...}, ...]  in RAM

# Step B: Create DataLoader
loader = DataLoader(dataset, batch_size=16, collate_fn=collate_episode_batch)
# loader knows: "I have 800 virtual items, give me 50 batches of 16"

# Step C: Training loop
for obs_b, act_b, rew_b, _ in loader:
    # obs_b: (T=25, B=16, 64, 64, 3)  float32 [0,1]
    # act_b: (T=25, B=16, action_dim)
    # rew_b: (T=25, B=16)
    loss = train_step(model, optimizer, obs_b, act_b, rew_b, device)

# Step D: Collect one new episode with the improved model
collect_experience(env, model, planner, storage, 1, config)
# Writes episode_000105.npz to Drive

# Step E: Optionally checkpoint
if (iteration + 1) % 10 == 0:
    save_checkpoint(model, optimizer, iteration, config)
    # Writes ckpt_000099.pt to Drive
```

---

## Summary

| Design decision | Reason |
|----------------|--------|
| Store episodes as `.npz uint8` on Drive | Survive Colab restarts; 4× smaller than float32 |
| Load all episodes into RAM each iteration | Avoid 800 slow Drive reads per iteration |
| `virtual_len = batch_size × steps` | Ensure full batches even with few episodes |
| `__getitem__` ignores index, samples randomly | Let the DataLoader manage batching, not sampling |
| Permute `(B,T,…)→(T,B,…)` in collate | Match `WorldModel.forward`'s time-first loop |
| Reconstruct `EpisodeDataset` each iteration | Pick up newly collected episodes automatically |
