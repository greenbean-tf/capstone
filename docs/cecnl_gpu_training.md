# CECNL 實驗室 GPU 訓練指南

## 環境說明

| 項目 | 說明 |
|------|------|
| 機器 | cecnl-Pro-ET700I-W7 |
| GPU | NVIDIA GeForce RTX 5090 (32GB) |
| Conda 路徑 | `/opt/anaconda3` |
| 環境名稱 | `capstone` |

---

## 第一次設定（只需做一次）

### 1. 初始化 conda

```bash
/opt/anaconda3/bin/conda init bash
source ~/.bashrc
```

### 2. 加入 CUDA 路徑

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3. 登入 Hugging Face

```bash
conda activate capstone
hf auth login --token <YOUR_HF_TOKEN>
```

---

## 每次登入的流程

### 1. 啟動環境

```bash
source ~/.bashrc
conda activate capstone
export HF_USER=greenbeanleo
```

### 2. 開啟 tmux（防止斷線中斷訓練）

> **注意：** conda 的 library 會與系統 tmux 衝突，必須加 `LD_LIBRARY_PATH=""` 才能正常執行 tmux 指令。

```bash
LD_LIBRARY_PATH="" tmux new -s train
```

### 3. 進入 tmux 後，重新啟動環境

```bash
source ~/.bashrc
conda activate capstone
export HF_USER=greenbeanleo
```

---

## 開始訓練

```bash
mkdir -p logs && \
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --dataset.repo_id=greenbeanleo/toyblock_synth_dataset \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --policy.type=diffusion \
  --output_dir=outputs/train/toyblocks_v8 \
  --job_name=toyblocks_v5 \
  --policy.device=cuda \
  --policy.repo_id=greenbeanleo/toyblocks_policy_v5 \
  --batch_size=64 \
  --steps=1000000 \
  --policy.n_obs_steps=2 \
  --policy.num_inference_steps=100 \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.crop_shape="[240,240]" \
  --wandb.enable=false \
  2>&1 | tee logs/toyblocks_v8.log
```

> **Log 說明：** `PYTHONUNBUFFERED=1` 停用 Python 的輸出緩衝，確保每行 log 即時寫入檔案（若缺少此設定，crash 時最後一段 log 可能遺失）。Log 存在 `logs/toyblocks_v8.log`，與 `output_dir` 分開放，避免和 lerobot-train 建立資料夾的時機衝突。

> **WandB 說明：** 實驗室網路封鎖 wandb.ai，請使用 `--wandb.enable=false`。
>
> **參數說明（相較舊版的改動）：**
>
> | 參數 | 舊值 | 新值 | 原因 |
> |------|------|------|------|
> | `batch_size` | 8 | 64 | 論文標準值，梯度估計更穩定 |
> | `steps` | 200,000 | 1,000,000 | 提升等效訓練 epoch 數 |
> | `dataset.image_transforms.enable` | false | true | 開啟色彩/仿射 augmentation |
> | `policy.n_obs_steps` | 3 | 2 | 論文 ablation 最佳值 |
> | `policy.num_inference_steps` | null | 100 | 明確設定 DDPM 推論步數 |
> | `policy.use_separate_rgb_encoder_per_camera` | false | true | wrist/front 雙攝影機分開 encoder |
> | `policy.crop_shape` | [84,84] | [240,240] | 覆蓋率從 2.3% 提升至 26%；`resize_shape` 雖更接近論文做法，但 rollout 環境（lerobot 0.4.2）會將其 strip 掉導致 train/test 不一致，故改用 crop_shape 直接裁切 |

### 離開 tmux（訓練繼續在背景跑）

按 **Ctrl+B**，然後按 **D**。

---

## 監控訓練

### 查看 GPU 狀態

```bash
gpustat
```

正常狀態：GPU 使用率 > 0%，記憶體有佔用，出現兩個 `green_bean` process（主程式 + dataloader）。

### 查看訓練 log

回到 tmux 看即時輸出：

```bash
LD_LIBRARY_PATH="" tmux attach -t train
```

或在另一個視窗追蹤 log 檔：

```bash
tail -f logs/toyblocks_v8.log
```

Log 格式說明：

```
step:84K  smpl:670K  ep:318  epch:39.72  loss:0.003  grdn:0.245  lr:6.5e-06
```

| 欄位 | 說明 |
|------|------|
| `step` | 目前訓練步數（總共 1,000,000 步）|
| `loss` | 損失值，越小越好 |
| `epch` | 目前 epoch 進度 |
| `lr` | 目前學習率 |

### 估算剩餘時間

每步約 0.1～0.15 秒（batch_size=64，雙攝影機分開 encoder），1,000,000 步總共約 **28～42 小時**。建議在 tmux 內執行並離開讓它在背景跑。

---

## 訓練結果

訓練完成後自動上傳到 HuggingFace：`https://huggingface.co/greenbeanleo/toyblocks_policy_v5`

本地結果存在：

```
outputs/train/toyblocks_v8/
├── checkpoints/        # 模型 checkpoint（每 20000 步存一次）
├── train_config.json   # 所有超參數
└── logs/               # 訓練 log
```

---

## 常用 tmux 指令

| 動作 | 指令 |
|------|------|
| 建立新 session | `LD_LIBRARY_PATH="" tmux new -s train` |
| 離開（訓練繼續） | Ctrl+B → D |
| 回到 session | `LD_LIBRARY_PATH="" tmux attach -t train` |
| 關閉 session | `LD_LIBRARY_PATH="" tmux kill-session -t train` |

---

## 注意事項

- 所有 tmux 指令前必須加 `LD_LIBRARY_PATH=""`，否則會因 conda library 衝突報錯
- `source ~/.bashrc` 後 conda 會回到 base，需重新執行 `conda activate capstone`
- `CUDA_VISIBLE_DEVICES=0` 指定使用 GPU 0，避免佔用其他人的資源，先用 `gpustat` 確認哪張空著
- 訓練結束後 GPU 記憶體自動釋放
