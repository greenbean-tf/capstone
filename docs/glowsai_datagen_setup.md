# GlowsAI Synthetic Data Generation Setup

Step-by-step guide for running Stage 3 (simulator synthetic data generation) on GlowsAI, based on actual setup experience.

## Prerequisites

- GlowsAI instance with RTX 4090 (24 GB VRAM)
- HuggingFace account with write access
- `object_poses.json` — either:
  - Output of Stage 2 SLAM pipeline uploaded to HuggingFace, **or**
  - Generated synthetically on the host (see Part 1 Step 6 — no real recordings needed)

---

## Part 1: First-Time Host Setup (GlowsAI)

Only needed the first time on a new GlowsAI instance.

### 1. Clone the repository

```bash
git clone https://github.com/greenbean-tf/capstone.git
cd capstone
```

### 2. Initialize submodules

```bash
make submodules
```

### 3. Install HuggingFace CLI

The system `pip` on Ubuntu 24.04 conflicts with debian-managed packages. Use `--break-system-packages --ignore-installed`:

```bash
apt-get update
pip3 install --break-system-packages --ignore-installed "huggingface_hub[cli]"
```

> If `pip3` is not found, install it first: `apt install -y python3-pip`. Run `apt-get update` before this as well to avoid 404 errors from a stale package list.

Verify:

```bash
hf --version
```

### 4. Login to HuggingFace (Host)

```bash
hf auth login --token <YOUR_HF_TOKEN>
export HF_USER=greenbeanleo
```

### 5. Download object_poses.json (UMI real data)

Skip this step if you plan to use synthetic poses (Step 6).

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

### 6. Generate synthetic object poses (no real recordings needed)

Use this instead of Step 5 when you want more data or don't have UMI recordings.

There are two modes:

**Normal mode** — blocks placed randomly across the workspace:

```bash
cd capstone

python3 scripts/datagen/generate_synthetic_poses.py \
    --num_episodes 200 \
    --seed 42 \
    --output data/synthetic/object_poses.json
```

**Near-box mode** — blocks placed near the storage box (recommended):

> ⚠️ TA hint: during evaluation, blocks are mostly placed close to the storage box. Use `--near_box` to generate training data that matches the evaluation scenario.

```bash
python3 scripts/datagen/generate_synthetic_poses.py \
    --num_episodes 200 \
    --seed 42 \
    --near_box \
    --output data/synthetic/object_poses.json
```

| Flag | Description |
|------|-------------|
| `--num_episodes` | Number of episodes to generate (200 → ~70 successes at 35% FSM rate) |
| `--seed` | Random seed for reproducibility |
| `--near_box` | Place blocks within 35 cm of the storage box (matches evaluation scenario) |
| `--output` | Output path (pass this to `--object_poses` in Part 3) |

| Mode | Workspace | Box distance |
|------|-----------|-------------|
| Normal | X=[0.05, 0.60] Y=[-0.45, -0.28] | ≥ 0.20 m (away from box) |
| `--near_box` | X=[0.35, 0.65] Y=[-0.55, -0.25] | 0.08–0.35 m (near box) |

The script runs in milliseconds on CPU — no GPU or Docker needed.
Pass `data/synthetic/object_poses.json` as `--object_poses` in Part 3 Step 2.

---

## Part 2: Launch Docker with tmux

Always use tmux before launching Docker so the session survives SSH disconnection.

### 1. Install tmux (Host)

```bash
apt-get install -y tmux
```

### 2. Start tmux session

```bash
cd capstone
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
export HF_USER=greenbeanleo
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
    --object_poses data/synthetic/object_poses.json \
    --step_hz 10000
```

> `--object_poses` points to the scene config file generated in Part 1 Step 6. `<dataset_repo_name>` is the HuggingFace dataset repo name you choose (e.g. `toy_blocks_collection`). These two names are independent — do not confuse them.

> `--step_hz 10000` removes the default 60 Hz real-time throttle and lets the GPU run as fast as possible. This gives roughly **2–3× speedup** with no impact on data quality.

**If you have a second object_poses.json** (e.g., a second synthetic batch or a different recording day), append with `--resume` after the first run completes:

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
    --resume \
    --step_hz 10000
```

Wait for `Replayed all N episodes. Exiting the app.` before stopping.

### 2b. Continue generation on a new session

If you closed the container or are on a fresh GlowsAI instance, the local lerobot cache is empty.
You must **download the existing dataset first** before using `--resume`, otherwise the script cannot determine how many episodes were already recorded.

> ⚠️ **Path distinction:** `data/synthetic/` is where your scene config (`object_poses.json`) lives. The lerobot cache at `/root/.cache/huggingface/lerobot/greenbeanleo/toyblock_synth_dataset` is where training data is stored. The `--local-dir` below **must end with `toyblock_synth_dataset`** — it must match exactly what you pass to `--lerobot_dataset_repo_id`.

**Step 1：下載現有資料集到 cache**

```bash
hf download greenbeanleo/toyblock_synth_dataset \
    --repo-type dataset \
    --revision v3.0 \
    --local-dir /root/.cache/huggingface/lerobot/greenbeanleo/toyblock_synth_dataset
```

**Step 2：生成新一批 synthetic poses**

```bash
python3 scripts/datagen/generate_synthetic_poses.py \
    --num_episodes 200 \
    --seed 123 \
    --near_box \
    --output data/synthetic_batch2/object_poses.json
```

> 每次接續生成時換一個 `--seed`（第一批用 42，第二批用 123，第三批用 456…），避免生成重複的物體位置。`--near_box` 讓積木生成在盒子附近，貼近評分情境。

**Step 3：接續生成**

```bash
python scripts/datagen/generate.py \
    --task HCIS-ToyBlocksCollection-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --use_lerobot_recorder \
    --lerobot_dataset_repo_id greenbeanleo/toyblock_synth_dataset \
    --object_poses data/synthetic_batch2/object_poses.json \
    --resume \
    --step_hz 10000
```

等出現 `Replayed all N episodes. Exiting the app.` 再繼續。

`--resume` 會讀 `/root/.cache/huggingface/lerobot/greenbeanleo/toyblock_synth_dataset/meta/info.json` 取得現有 episode 數，從最後一筆繼續接。

**Step 4：上傳合併後的完整資料集**

```bash
hf upload greenbeanleo/toyblock_synth_dataset \
    /root/.cache/huggingface/lerobot/greenbeanleo/toyblock_synth_dataset/ \
    --repo-type dataset
```

> `generate.py` 結束時會自動更新 HuggingFace 上的 `v3.0` tag，指向最新的 commit。不需要手動更新 tag。

**Step 5：回 cecnl 機器訓練前，清除舊 cache**

```bash
rm -rf ~/.cache/huggingface/lerobot/greenbeanleo/toyblock_synth_dataset
```

這樣 lerobot-train 才會重新下載包含新 episodes 的完整資料集。

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

### `--resume` bugs — manual patch required

Two bugs in `leisaac.enhance.datasets.lerobot_dataset_handler` affect `--resume` mode:

1. `clear()` crashes with `TypeError: 'NoneType' object is not subscriptable` when `episode_buffer` is `None` on the first reset.
2. `get_num_episodes()` raises `NotImplementedError`.

These are fixed in `Dockerfile`, but if your container image was built before the patch was added (Docker layer cache), the fix is not applied. **Run this inside the container every time you start a new session:**

```bash
python3 -c "
path = '/usr/local/lib/python3.11/dist-packages/leisaac/enhance/datasets/lerobot_dataset_handler.py'
content = open(path).read()
content = content.replace(
    '    def clear(self):\n        self._lerobot_dataset.clear_episode_buffer()',
    '    def clear(self):\n        if self._lerobot_dataset.episode_buffer is None:\n            return\n        self._lerobot_dataset.clear_episode_buffer()'
)
content = content.replace(
    '    def get_num_episodes(self) -> int:\n        raise NotImplementedError(\"get_num_episodes is not supported for LeRobotDatasetHandler\")',
    '    def get_num_episodes(self) -> int:\n        return self._lerobot_dataset.num_episodes'
)
open(path, 'w').write(content)
print('Both patches applied OK')
"
```

To permanently fix, rebuild the image from the host (outside the container):

```bash
docker build --no-cache -f Dockerfile -t leisaac-isaaclab:latest .
```

This takes 20–60 minutes but only needs to be done once.

---

## Expected Results

| Scenario | Typical outcome |
|----------|----------------|
| `status=full` poses with blocks in robot workspace | ~30–50% success rate |
| Synthetic poses (from `generate_synthetic_poses.py`) | ~35% success rate; 200 episodes → ~70 successes |
| `status=full` poses with blocks far outside workspace (Y ≈ 0) | 0% success — all episodes fail |
| `status=none` poses | Skipped automatically (no object data) |

**Recommended dataset size:**

| Policy | Minimum | Recommended |
|--------|---------|-------------|
| Diffusion Policy | 50 episodes | 70–150 episodes |
| ACT | 20 episodes | 50+ episodes |

8–20 episodes may be sufficient for a quick proof-of-concept but diffusion policy typically needs **50+ episodes** to generalise well. Use `generate_synthetic_poses.py` with `--num_episodes 200` to reach this target in one run.
