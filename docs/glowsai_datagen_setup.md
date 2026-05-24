# GlowsAI Synthetic Data Generation Setup

Step-by-step guide for running Stage 3 (simulator synthetic data generation) on GlowsAI, based on actual setup experience.

## Prerequisites

- GlowsAI instance with RTX 4090 (24 GB VRAM)
- HuggingFace account with write access
- `object_poses.json` uploaded to HuggingFace (output of Stage 2 SLAM pipeline)

---

## Part 1: First-Time Host Setup (GlowsAI)

Only needed the first time on a new GlowsAI instance.

### 1. Clone the repository

```bash
git clone https://github.com/greenbean-tf/capstone.git
cd ~/Desktop/capstone
```

### 2. Initialize submodules

```bash
make submodules
```

### 3. Install HuggingFace CLI

The system `pip` on Ubuntu 24.04 conflicts with debian-managed packages. Use `--break-system-packages --ignore-installed`:

```bash
pip3 install --break-system-packages --ignore-installed "huggingface_hub[cli]"
```

Verify:

```bash
hf --version
```

### 4. Login to HuggingFace (Host)

```bash
hf auth login --token <YOUR_HF_TOKEN>
export HF_USER=<your-huggingface-username>
```

### 5. Download object_poses.json

```bash
# Download from classmate's or your own HF repo
hf download <hf_user>/<repo_name> \
    object_poses.json \
    --local-dir data/<demo_directory_name>
```

Example for toy blocks collection:

```bash
hf download yujieee0616/toyblock_dataset \
    object_poses.json \
    --local-dir data/toyblock_dataset

hf download yujieee0616/toyblock_dataset_day1 \
    object_poses.json \
    --local-dir data/toyblock_dataset_day1
```

---

## Part 2: Launch Docker with tmux

Always use tmux before launching Docker so the session survives SSH disconnection.

### 1. Install tmux (Host)

```bash
apt-get install -y tmux
```

### 2. Start tmux session

```bash
cd ~/Desktop/capstone
tmux new -s capstone
```

Useful tmux shortcuts:
- `Ctrl+B, D` → detach (leave session running in background)
- `Ctrl+B, C` → new window
- `Ctrl+B, 0/1` → switch windows
- `tmux attach -t capstone` → re-attach after SSH reconnect

### 3. Launch Docker container (inside tmux)

```bash
make launch-isaaclab-glowsai-4090   # RTX 4090
# or
make launch-isaaclab-glowsai-l40s   # L40S
```

> First build takes 20–60 minutes. Subsequent launches reuse the cached image and start in ~2 minutes.

When you see `root@xxxx:/workspace/aicapstone#`, you are inside the container.

---

## Part 3: Inside Docker Container

All commands below run **inside the Docker container**.

### 1. Login to HuggingFace (Container)

The container has its own filesystem — you must login again inside:

```bash
hf auth login --token <YOUR_HF_TOKEN>
export HF_USER=<your-huggingface-username>
```

### 2. Run data generation

Available tasks:
- `HCIS-ToyBlocksCollection-SingleArm-v0`
- `HCIS-CupStacking-SingleArm-v0`
- `HCIS-CutleryArrangement-SingleArm-v0`

> ⚠️ **Always use `--num_envs 1`** for these tasks. The front camera (`/World/front_camera`) is a single global tile — using `--num_envs > 1` causes a CUDA device-side assert during environment reset.

**First run:**

```bash
python scripts/datagen/generate.py \
    --task HCIS-ToyBlocksCollection-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --use_lerobot_recorder \
    --lerobot_dataset_repo_id ${HF_USER}/<dataset_repo_name> \
    --object_poses data/<demo_directory_name>/object_poses.json
```

**If you have a second object_poses.json** (e.g., recorded on a different day), append with `--resume` after the first run completes:

```bash
python scripts/datagen/generate.py \
    --task HCIS-ToyBlocksCollection-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --use_lerobot_recorder \
    --lerobot_dataset_repo_id ${HF_USER}/<dataset_repo_name> \
    --object_poses data/<demo_directory_name_2>/object_poses.json \
    --resume
```

Wait for `Replayed all N episodes. Exiting the app.` before stopping.

### 3. Monitor VRAM (optional, in a second tmux window)

```bash
# Ctrl+B, C to open new tmux window
watch -n 2 nvidia-smi
# Expected: ~6 GB used with --num_envs 1
```

### 4. Upload dataset to HuggingFace

```bash
hf upload ${HF_USER}/<dataset_repo_name> \
    /root/.cache/huggingface/lerobot/${HF_USER}/<dataset_repo_name>/ \
    --repo-type dataset
```

---

## Understanding the Output

```
[Data Usage] 7/20 fail.
Start Recording!!!
Episode failed!
Stop Recording!!!
```

| Field | Meaning |
|-------|---------|
| `7` | Number of successful episodes recorded so far |
| `20` | Total `status=full` episodes in `object_poses.json` |
| `fail` | This episode failed (state machine could not complete the task) |

- Each episode is tried **exactly once** — failed episodes are skipped, not retried.
- The counter only increments on success.
- The system exits automatically after all episodes are attempted.
- **Do not Ctrl+C** while it is running — wait for the exit message.

If you accidentally Ctrl+C, restart with `--resume` to continue from where you left off.

---

## Known Issues & Fixes

### `FileExistsError` on restart

```
FileExistsError: [Errno 17] File exists: '/root/.cache/huggingface/lerobot/...'
```

**Fix:** Delete the leftover directory before restarting without `--resume`:

```bash
rm -rf /root/.cache/huggingface/lerobot/${HF_USER}/<dataset_repo_name>
```

### `--num_envs > 1` crashes with CUDA device-side assert

The `front` camera is defined at `/World/front_camera` (a single global tile). When multiple environments are reset simultaneously, the camera's `_timestamp_last_update` tensor is indexed out of bounds.

**Fix:** Always use `--num_envs 1` for these tasks.

### `--resume` bugs (already patched in Dockerfile)

Two bugs in `leisaac.enhance.datasets.lerobot_dataset_handler` affect `--resume` mode:

1. `clear()` crashes with `TypeError: 'NoneType' object is not subscriptable` when `episode_buffer` is `None` on the first reset.
2. `get_num_episodes()` raises `NotImplementedError`.

Both are fixed in `Dockerfile` via a post-install patch. No manual action needed as long as you use the Docker image built from this repo.

---

## Expected Results

| Scenario | Typical outcome |
|----------|----------------|
| `status=full` poses with blocks in robot workspace | ~30–50% success rate |
| `status=full` poses with blocks far outside workspace (Y ≈ 0) | 0% success — all episodes fail |
| `status=none` poses | Skipped automatically (no object data) |

A dataset with **8–20 successful episodes** is sufficient to proceed to Stage 4 (LeRobot training).
