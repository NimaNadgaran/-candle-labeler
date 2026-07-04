"""
model_agent.py
=================================================================
The online-learning model that watches the human trader's decisions
and gradually learns to predict buy/sell/neutral for each of the 6
lookback timeframes (300/200/100/50/20/10 candles).

Called from main.py in two places per batch:
  agent.predict_batch(df_charts, random_number)
      -> returns the model's current buy/sell/neutral guess for each
         timeframe, BEFORE it sees the human's actual choice. Purely
         for comparison/logging - does not affect training.
  agent.update_model(df_charts, random_number, human_choices)
      -> trains the model on the human's labeled decision and
         auto-saves the updated model to disk.

How a single decision point becomes a training example:
  1. Feature extraction (_extract_features): each lookback window is
     summarized into a 7-number "shape" description (return,
     volatility, where price sits in its range, etc.) rather than a
     single flat average - see the docstring on _extract_features for
     the full list.
  2. Normalization (RunningNormalizer): those 7 numbers are
     standardized using a running mean/variance per timeframe, so
     absolute price level (e.g. gold at 2000 vs 3000) never dominates.
  3. Replay buffer: every labeled example is stored to disk
     (model_dir/replay_buffer.json, capped at REPLAY_BUFFER_MAX). Each
     training step samples a mini-batch from this buffer (always
     including the newest example) instead of training on just the one
     newest label - this is what keeps the model learning general
     patterns instead of overfitting/forgetting on each single round.
  4. Everything (model weights, optimizer state, per-timeframe
     normalizer stats) is checkpointed to model_dir/online_model.pth
     after every update, so progress survives restarts.
=================================================================
"""

import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Map string actions to numerical labels for PyTorch CrossEntropyLoss.
# Both spellings map to the same "neutral" class so historical data / the
# main.py fallback string ("neurall") and the chart_app.py button label
# ("neutral") are both handled explicitly instead of relying on a coincidence.
LABEL_MAP = {"buy": 0, "sell": 1, "neutral": 2, "neurall": 2}
INV_LABEL_MAP = {0: "buy", 1: "sell", 2: "neutral"}
TIMEFRAMES = [300, 200, 100, 50, 20, 10]

FEATURE_DIM = 7  # see _extract_features for what each dimension means
REPLAY_BUFFER_MAX = 500
BATCH_SIZE = 16


class MultiFramePredictor(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM):
        super(MultiFramePredictor, self).__init__()
        # Shared feature extraction layers (processes individual candle feature vectors)
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 6 Separate output heads, one for each timeframe lookback horizon
        # Each head outputs 3 raw scores (logits) representing [buy, sell, neutral]
        self.heads = nn.ModuleDict({
            str(tf): nn.Linear(32, 3) for tf in TIMEFRAMES
        })

    def forward(self, x_dict):
        """
        x_dict: A dictionary where keys are timeframes (str) and values are
                the tensor features (already normalized) for that frame view,
                shape [batch, FEATURE_DIM].
        """
        outputs = {}
        for tf in TIMEFRAMES:
            tf_str = str(tf)
            shared_feat = self.shared_net(x_dict[tf_str])
            outputs[tf_str] = self.heads[tf_str](shared_feat)
        return outputs


class RunningNormalizer:
    """
    Per-feature online mean/variance tracker using an exponential moving
    average. Keeps a separate normalizer per timeframe since the statistical
    profile of a 300-candle window differs from a 10-candle window.
    Persisted alongside the model checkpoint so normalization stays stable
    across restarts.
    """

    def __init__(self, dim=FEATURE_DIM, momentum=0.02):
        self.dim = dim
        self.momentum = momentum
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.initialized = False

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if not self.initialized:
            self.mean = x.copy()
            self.var = np.ones_like(x)
            self.initialized = True
        else:
            delta = x - self.mean
            self.mean = self.mean + self.momentum * delta
            self.var = (1 - self.momentum) * self.var + self.momentum * (delta ** 2)

    def normalize(self, x):
        x = np.asarray(x, dtype=np.float64)
        std = np.sqrt(self.var) + 1e-6
        return (x - self.mean) / std

    def state_dict(self):
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "initialized": self.initialized}

    def load_state_dict(self, state):
        self.mean = np.array(state["mean"], dtype=np.float64)
        self.var = np.array(state["var"], dtype=np.float64)
        self.initialized = state.get("initialized", True)


class OnlineModelAgent:
    def __init__(self, model_dir="model_dir", lr=0.005):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "online_model.pth")
        self.buffer_path = os.path.join(model_dir, "replay_buffer.json")

        self.model = MultiFramePredictor(input_dim=FEATURE_DIM)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        # One running normalizer per timeframe head
        self.normalizers = {tf: RunningNormalizer(FEATURE_DIM) for tf in TIMEFRAMES}

        # Replay buffer: list of {"features": {tf_str: [..]}, "choices": {tf_str: "buy"}}
        self.replay_buffer = []

        # Auto-initialize directory and load existing weights if available
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            print(f"Created directory '{self.model_dir}'. Initializing fresh model...")
            self.save_model()
        else:
            self.load_model()
            self.load_replay_buffer()

    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'normalizers': {str(tf): self.normalizers[tf].state_dict() for tf in TIMEFRAMES},
        }, self.model_path)

    def load_model(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'normalizers' in checkpoint:
                    for tf in TIMEFRAMES:
                        tf_str = str(tf)
                        if tf_str in checkpoint['normalizers']:
                            self.normalizers[tf].load_state_dict(checkpoint['normalizers'][tf_str])
                print("Successfully loaded existing model and optimizer states from disk.")
            except RuntimeError as e:
                # Most likely cause: input_dim (feature count) changed since this
                # checkpoint was saved, so shapes no longer match. Rather than
                # crash, start fresh with a clear warning.
                print("WARNING: Existing checkpoint is incompatible with the current "
                      f"model architecture ({e}). Starting from a freshly initialized model.")
        else:
            print("No model snapshot found. Running on initialized weights.")

    def load_replay_buffer(self):
        if os.path.exists(self.buffer_path):
            try:
                with open(self.buffer_path, "r") as f:
                    self.replay_buffer = json.load(f)
                print(f"Loaded {len(self.replay_buffer)} past labeled samples from replay buffer.")
            except (json.JSONDecodeError, OSError):
                print("WARNING: Replay buffer file was unreadable, starting with an empty buffer.")
                self.replay_buffer = []

    def save_replay_buffer(self):
        with open(self.buffer_path, "w") as f:
            json.dump(self.replay_buffer, f)

    def _extract_features(self, df_charts, random_number, timeframe):
        """
        Extracts a 7-dim feature vector describing the shape of the price
        window, instead of just its flat OHLCV mean:

          0: total_return       - overall move across the window
          1: mean_return        - average per-candle return (drift)
          2: volatility         - std of per-candle returns
          3: range_ratio        - (max high - min low) relative to price level
          4: close_position     - where the last close sits within the window's range [0,1]
          5: body_ratio_mean    - average candle body size relative to its range
          6: volume_trend       - second-half vs first-half average volume, relative change

        Includes the current candle (random_number) itself, not just candles
        strictly before it.
        """
        start_idx = max(0, random_number - timeframe)
        end_idx = random_number + 1  # inclusive of the current candle

        slice_df = df_charts.iloc[start_idx:end_idx]
        eps = 1e-8

        if len(slice_df) < 2 or "close" not in slice_df.columns:
            return np.zeros(FEATURE_DIM, dtype=np.float64)

        opens = slice_df["open"].to_numpy(dtype=np.float64) if "open" in slice_df.columns else None
        highs = slice_df["high"].to_numpy(dtype=np.float64) if "high" in slice_df.columns else None
        lows = slice_df["low"].to_numpy(dtype=np.float64) if "low" in slice_df.columns else None
        closes = slice_df["close"].to_numpy(dtype=np.float64)

        total_return = (closes[-1] - closes[0]) / (closes[0] + eps)

        per_candle_returns = np.diff(closes) / (closes[:-1] + eps)
        mean_return = per_candle_returns.mean()
        volatility = per_candle_returns.std()

        if highs is not None and lows is not None:
            window_high = highs.max()
            window_low = lows.min()
            range_ratio = (window_high - window_low) / (closes.mean() + eps)
            close_position = (closes[-1] - window_low) / (window_high - window_low + eps)
        else:
            range_ratio = 0.0
            close_position = 0.5

        if opens is not None and highs is not None and lows is not None:
            candle_ranges = highs - lows
            body_ratio_mean = np.mean(np.abs(closes - opens) / (candle_ranges + eps))
        else:
            body_ratio_mean = 0.0

        if "volume" in slice_df.columns:
            volumes = slice_df["volume"].to_numpy(dtype=np.float64)
            half = max(1, len(volumes) // 2)
            first_half_mean = volumes[:half].mean()
            second_half_mean = volumes[half:].mean() if len(volumes[half:]) > 0 else first_half_mean
            volume_trend = (second_half_mean - first_half_mean) / (first_half_mean + eps)
        else:
            volume_trend = 0.0

        return np.array([
            total_return, mean_return, volatility, range_ratio,
            close_position, body_ratio_mean, volume_trend
        ], dtype=np.float64)

    def _extract_all_features(self, df_charts, random_number):
        """Returns {tf_str: raw_feature_vector (np.ndarray, dim FEATURE_DIM)} for all timeframes."""
        return {str(tf): self._extract_features(df_charts, random_number, tf) for tf in TIMEFRAMES}

    def _to_normalized_tensor_dict(self, raw_features_dict, update_normalizer=False):
        """Converts {tf_str: raw_vector} into {tf_str: tensor[1, FEATURE_DIM]}, normalized."""
        x_dict = {}
        for tf in TIMEFRAMES:
            tf_str = str(tf)
            raw = np.asarray(raw_features_dict[tf_str], dtype=np.float64)
            if update_normalizer:
                self.normalizers[tf].update(raw)
            normed = self.normalizers[tf].normalize(raw)
            x_dict[tf_str] = torch.tensor(normed, dtype=torch.float32).unsqueeze(0)
        return x_dict

    def predict_batch(self, df_charts, random_number):
        """Step 5: Predict buy/sell/neutral across all 6 lookbacks before updating"""
        self.model.eval()
        raw_features = self._extract_all_features(df_charts, random_number)
        x_dict = self._to_normalized_tensor_dict(raw_features, update_normalizer=False)
        predictions = {}

        with torch.no_grad():
            outputs = self.model(x_dict)
            for tf in TIMEFRAMES:
                logits = outputs[str(tf)]
                pred_idx = torch.argmax(logits, dim=1).item()
                predictions[tf] = INV_LABEL_MAP[pred_idx]

        return predictions

    def update_model(self, df_charts, random_number, human_choices):
        """
        Step 6 & 7: Learn on human validation tags and checkpoint instantly.

        Instead of training on only this single newest example (which causes
        the model to overfit/forget on each round), the new example is added
        to a persistent replay buffer and a mini-batch is sampled from it
        (including the newest point) for the actual gradient step. This is
        what makes the loss curve reflect real learning progress rather than
        single-sample memorization.
        """
        raw_features = self._extract_all_features(df_charts, random_number)

        # Update running normalizers with this new sample's statistics.
        for tf in TIMEFRAMES:
            self.normalizers[tf].update(raw_features[str(tf)])

        # Store the new labeled example (JSON-safe: plain lists/strings).
        entry = {
            "features": {tf_str: vec.tolist() for tf_str, vec in raw_features.items()},
            "choices": {str(tf): str(human_choices.get(tf, "neutral")) for tf in TIMEFRAMES},
        }
        self.replay_buffer.append(entry)
        if len(self.replay_buffer) > REPLAY_BUFFER_MAX:
            self.replay_buffer = self.replay_buffer[-REPLAY_BUFFER_MAX:]

        # Build a mini-batch: the newest sample plus a random sample of past ones.
        batch_size = min(BATCH_SIZE, len(self.replay_buffer))
        if len(self.replay_buffer) <= batch_size:
            batch = self.replay_buffer
        else:
            batch = [self.replay_buffer[-1]] + random.sample(self.replay_buffer[:-1], batch_size - 1)

        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0
        for tf in TIMEFRAMES:
            tf_str = str(tf)

            batch_features = np.stack([
                self.normalizers[tf].normalize(np.asarray(sample["features"][tf_str], dtype=np.float64))
                for sample in batch
            ])
            batch_targets = [
                LABEL_MAP.get(sample["choices"].get(tf_str, "neutral"), 2)
                for sample in batch
            ]

            x_batch = torch.tensor(batch_features, dtype=torch.float32)
            y_batch = torch.tensor(batch_targets, dtype=torch.long)

            logits = self.model.shared_net(x_batch)
            logits = self.model.heads[tf_str](logits)

            loss = self.criterion(logits, y_batch)
            total_loss += loss

        total_loss.backward()
        self.optimizer.step()

        # Checkpoint model + normalizers + replay buffer to disk.
        self.save_model()
        self.save_replay_buffer()

        return total_loss.item()