# model_dir / README

This directory contains the **trained model files** and training memory for the Candle Labeler project.

It is automatically created and managed by `model_agent.py`.

---

## Directory Contents

| File                    | Description                                                                 | Important? |
|-------------------------|-----------------------------------------------------------------------------|------------|
| `online_model.pth`      | Main PyTorch model checkpoint (weights + optimizer state)                   | Yes        |
| `replay_buffer.json`    | Persistent memory of all your past labeling decisions                       | Yes        |

---

## 1. `online_model.pth`

**Type:** PyTorch checkpoint (`.pth`)

**Contains:**
- Model architecture (`MultiFramePredictor`)
- Trained weights for all 6 timeframe heads
- Optimizer state (Adam optimizer)
- Running normalizer statistics (mean & variance per feature per timeframe)
- Training metadata

**Purpose:**
This file allows the model to **remember what it has learned** from your labeling sessions. It is automatically loaded when you start `main.py` and updated after every batch.

---

## 2. `replay_buffer.json`

**Type:** JSON file

**Purpose:**
Stores a history of your past labeling decisions (up to 500 samples). This replay buffer is crucial for stable online learning — it prevents the model from forgetting old patterns and helps it generalize better.

**Structure of each entry:**
```json
{
  "features": {
    "300": [feature_vector],
    "200": [feature_vector],
    ...
  },
  "choices": {
    "300": "buy",
    "200": "neutral",
    ...
  }
}
```
## How It Works

1. When you run `main.py` for the first time:
- `model_dir/` is created automatically
- A fresh model is initialized and saved

2. After each labeling batch:
- Your decisions are added to the replay buffer
- The model trains on a mini-batch sampled from the buffer
- Both the model and replay buffer are auto-saved

3. On restart:
- The model and normalizer statistics are restored
- Training continues from where you left off



## Best Practices

- Do not delete these files unless you want to start training from scratch.
- Backup this folder regularly if you have spent many hours labeling.
- The more you label, the smarter the model becomes.
- You can safely copy this folder to another machine to continue training.


## Technical Details

- Model Type: Multi-head neural network with shared backbone
- Input Features: 7 hand-crafted technical features per timeframe
- Normalization: Online running normalization (per timeframe)
- Training Method: Online learning with replay buffer
- Loss Function: CrossEntropyLoss
- Optimizer: Adam (lr = 0.005)


## Resetting the Model
If you want to start fresh:
```Bash
# Delete the model directory (model will be reinitialized on next run)
rm -rf model_dir/
```
Then run python main.py again.

This directory is the "brain" of your Candle Labeler project.
The more quality labels you provide, the more accurate and personalized the model becomes.
Happy Training! 🤖
