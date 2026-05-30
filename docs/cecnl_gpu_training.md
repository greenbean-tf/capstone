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
CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --dataset.repo_id=greenbeanleo/toyblock_synth_dataset \
  --dataset.video_backend=pyav \
  --policy.type=diffusion \
  --output_dir=outputs/train/toyblocks_v1 \
  --job_name=toyblocks_v1 \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=greenbeanleo/toyblocks_policy
```

> **WandB 說明：** 實驗室網路封鎖 wandb.ai，請使用 `--wandb.enable=false`。

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

回到 tmux：

```bash
LD_LIBRARY_PATH="" tmux attach -t train
```

Log 格式說明：

```
step:84K  smpl:670K  ep:318  epch:39.72  loss:0.003  grdn:0.245  lr:6.5e-06
```

| 欄位 | 說明 |
|------|------|
| `step` | 目前訓練步數（總共 100,000 步）|
| `loss` | 損失值，越小越好 |
| `epch` | 目前 epoch 進度 |
| `lr` | 目前學習率 |

### 估算剩餘時間

每步約 0.045 秒，100,000 步總共約 **75 分鐘**。

---

## 訓練結果

訓練完成後自動上傳到 HuggingFace：`https://huggingface.co/greenbeanleo/toyblocks_policy`

本地結果存在：

```
outputs/train/toyblocks_v1/
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
