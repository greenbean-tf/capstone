# ColorSortBlocks Advanced Level

完整流程：資料生成 → 訓練 → 評估。

---

## 任務說明

**Task ID：** `HCIS-ColorSortBlocks-SingleArm-v0`

機器人需將三個顏色積木分別放入對應顏色的籃子：

| 積木 | 目標籃子 | 籃子位置 (world) |
|------|----------|-----------------|
| green_block | green_basket | (0.65, -0.55) |
| blue_block  | blue_basket  | (0.35, -0.55) |
| red_block   | red_basket   | (0.05, -0.55) |

成功條件：三個積木**同時**全部在正確籃子內（±12 cm x/y，±8 cm z）。

### Advanced Level 與 Entry Level 的差異

| | Entry Level | Advanced Level |
|--|-------------|----------------|
| 積木起始位置 | 固定座標 | 隨機採樣 |
| 迴避範圍 | 單一籃子 | **三個籃子全部**迴避 |
| 工作空間 Y 範圍 | 較寬 | Y=[-0.36, -0.15]（靠近籃子） |
| 難度 | 低 | 高 |

---

## Part 1：資料生成（GlowsAI）

### 1. 啟動 Docker

```bash
cd capstone
tmux new -s capstone
make launch-isaaclab-glowsai-4090   # RTX 4090
```

### 2. 登入 HuggingFace（Container 內）

```bash
hf auth login --token <YOUR_HF_TOKEN>
export HF_USER=greenbeanleo
```

### 3. 下載現有資料集

若 Container 是新開的，local cache 是空的，需先下載：

```bash
hf download greenbeanleo/color_sort_blocks_dataset \
    --repo-type dataset \
    --local-dir /root/.cache/huggingface/lerobot/greenbeanleo/color_sort_blocks_dataset
```

> ⚠️ `--local-dir` 最後一層資料夾名稱必須與 repo_id 最後一段一致（`color_sort_blocks_dataset`）。

### 4. 生成 Advanced Level Poses

```bash
python3 scripts/datagen/generate_synthetic_poses_advanced.py \
    --num_episodes 1200 \
    --seed 123 \
    --output data/advanced_batch2/object_poses.json
```

| 參數 | 說明 |
|------|------|
| `--num_episodes` | 總 episode 數（包含 FSM 失敗的）|
| `--seed` | 隨機種子，每批換一個避免重複 |
| `--output` | 輸出路徑，傳給 generate.py 的 `--object_poses` |

工作空間：X=[0.05, 0.60]、Y=[-0.36, -0.15]（world frame），積木間距 ≥ 0.15 m，所有三個籃子位置都會排除。

### 5. 接續生成資料

```bash
python scripts/datagen/generate.py \
    --task HCIS-ColorSortBlocks-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --use_lerobot_recorder \
    --lerobot_dataset_repo_id greenbeanleo/color_sort_blocks_dataset \
    --object_poses data/advanced_batch2/object_poses.json \
    --resume \
    --step_hz 10000
```

> `--resume` 從本地 cache 讀取現有 episode 數，從斷點繼續接。  
> 等出現 `Replayed all N episodes. Exiting the app.` 後再繼續。

> ⚠️ 若出現 VNC + Vulkan 的 `backbuffers are not initialized` 錯誤，加 `--headless` 參數。

### 6. 上傳資料集 & 更新 Tag

```bash
hf upload greenbeanleo/color_sort_blocks_dataset \
    /root/.cache/huggingface/lerobot/greenbeanleo/color_sort_blocks_dataset/ \
    --repo-type dataset
```

```python
python3 << 'EOF'
from huggingface_hub import HfApi
api = HfApi()
try:
    api.delete_tag("greenbeanleo/color_sort_blocks_dataset", tag="v3.0", repo_type="dataset")
except:
    pass
api.create_tag("greenbeanleo/color_sort_blocks_dataset", tag="v3.0", repo_type="dataset")
print("v3.0 tag updated")
EOF
```

---

## Part 2：訓練 ACT Policy（CECNL）

在 cecnl 機器上，使用 conda 環境訓練。

### 1. 準備環境

```bash
source ~/.bashrc
conda activate capstone
export HF_USER=greenbeanleo
LD_LIBRARY_PATH="" tmux new -s train
# tmux 內重新 activate
source ~/.bashrc
conda activate capstone
```

### 2. 訓練指令

```bash
mkdir -p logs && \
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --dataset.repo_id=greenbeanleo/color_sort_blocks_dataset \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.repo_id=greenbeanleo/color_sort_blocks_act_policy_v1 \
  --output_dir=outputs/train/color_sort_blocks_act_v1 \
  --job_name=color_sort_blocks_act_v1 \
  --batch_size=8 \
  --steps=100000 \
  --wandb.enable=false \
  2>&1 | tee logs/color_sort_blocks_act_v1.log
```

### 訓練參數說明

| 參數 | 值 | 依據 |
|------|-----|------|
| `policy.type` | act | Diffusion 在 1.16 epoch 成功率為 0；ACT 收斂更快 |
| `batch_size` | 8 | ACT 論文（Zhao et al., 2023）Table III 原始設定 |
| `chunk_size` | 100（預設） | 論文 ablation 最佳值 |
| `kl_weight` | 10（預設） | 論文設定，無 ablation 依據不調整 |
| `steps` | 100000 | 起始 baseline；視成功率決定是否加到 300K |

> ⚠️ `--policy.repo_id` 只有 ACT 才需要此格式，Diffusion 的參數名稱相同。若出現 `ParsingError: Couldn't instantiate class ACTConfig`，檢查是否有非法的 `--policy.*` 參數（如 `batch_size` > 8）。

### 監控訓練

```bash
# 查看即時 log
LD_LIBRARY_PATH="" tmux attach -t train

# 另一視窗追蹤
tail -f logs/color_sort_blocks_act_v1.log
```

Log 欄位說明：

```
step:50K  smpl:400K  ep:200  epch:0.58  loss:0.012  grdn:0.45  lr:5.0e-05
```

| 欄位 | 說明 |
|------|------|
| `epch` | 等效 epoch（100K steps ≈ 1.16 epoch） |
| `loss` | 應穩定下降至 < 0.01 |
| `grdn` | gradient norm，正常 < 1.0 |

### 若 100K 不夠，繼續訓練

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --dataset.repo_id=greenbeanleo/color_sort_blocks_dataset \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.repo_id=greenbeanleo/color_sort_blocks_act_policy_v2 \
  --output_dir=outputs/train/color_sort_blocks_act_v2 \
  --job_name=color_sort_blocks_act_v2 \
  --batch_size=8 \
  --steps=300000 \
  --wandb.enable=false \
  2>&1 | tee logs/color_sort_blocks_act_v2.log
```

---

## Part 3：評估（GlowsAI Docker）

### 1. 下載訓練好的 Policy

```bash
hf download greenbeanleo/color_sort_blocks_act_policy_v1 \
    --local-dir checkpoints/color_sort_blocks_act_policy_v1
```

### 2. 執行評估

**固定位置（基本測試）：**

```bash
python scripts/rollout.py \
    --task HCIS-ColorSortBlocks-SingleArm-v0 \
    --policy_type lerobot-act \
    --policy_checkpoint_path checkpoints/color_sort_blocks_act_policy_v1 \
    --policy_action_horizon 100 \
    --episode_length_s 30 \
    --eval_rounds 20 \
    --device cuda \
    --enable_cameras
```

**隨機位置（與訓練資料分布一致）：**

```bash
python scripts/rollout.py \
    --task HCIS-ColorSortBlocks-SingleArm-v0 \
    --policy_type lerobot-act \
    --policy_checkpoint_path checkpoints/color_sort_blocks_act_policy_v1 \
    --policy_action_horizon 100 \
    --episode_length_s 30 \
    --eval_rounds 20 \
    --device cuda \
    --enable_cameras \
    --object_poses data/advanced_batch2/object_poses.json
```

> `--object_poses` 省略時維持固定初始位置（向下相容）。  
> 指定後，每輪 episode 依序套用 JSON 的 pose，超過總數後循環。

### 3. 評估輸出

每個 episode 結束時：

```
[Evaluation] Episode 3 successful! Time: 18.4s | green=✓ blue=✓ red=✓
[Evaluation] Episode 7 timed out!              | green=✓ blue=✗ red=✗
[Evaluation] Running: full=1/7 | green=5/7 | blue=3/7 | red=2/7
```

最終彙總：

```
[Evaluation] ========== Final Results ==========
  Full success rate  : 3/20 (15.0%)
  Green block in box : 10/20 (50.0%)
  Blue  block in box : 6/20  (30.0%)
  Red   block in box : 4/20  (20.0%)
  Avg completion time: 21.3s (over 3 successful episodes)
[Evaluation] ========================================
```

| 指標 | 說明 |
|------|------|
| Full success rate | 三個積木全部放對才算成功 |
| Per-color success | 個別積木的放置成功率（可診斷哪個顏色最難） |
| Avg completion time | 完整成功的 episode 平均花了幾秒 |

---

## 常見問題

### `RevisionNotFoundError`

```
RevisionNotFoundError: dataset must be tagged with v3.0
```

**原因：** HuggingFace 上的 dataset 沒有 `v3.0` tag。  
**修復：** 執行 Part 1 Step 6 的 tag 更新指令。

### `backbuffers are not initialized`

**原因：** VNC + Vulkan 不相容。  
**修復：** 加 `--headless` 到 generate.py 指令。

### `ParsingError: Couldn't instantiate class ACTConfig`

**原因：** 傳入了 ACTConfig 不支援的 CLI 參數（已知 `batch_size > 8` 會觸發）。  
**修復：** 使用 `--batch_size=8`。

### 積木掉出桌面

**原因：** `num_envs > 1` 時物體初始化沒有加上 env_origins offset。  
**修復：** 永遠使用 `--num_envs 1`。
