# Candle Labeler

An interactive **human-in-the-loop** labeling tool for training a multi-timeframe trading model on candlestick charts.

You label price action across 6 different lookback windows (300, 200, 100, 50, 20, 10 candles), and an online-learning neural network learns from your decisions in real time.

---

## Features

- **Interactive Dash + Plotly** candlestick charts with volume profile
- Click anywhere on chart to draw horizontal reference lines
- 6 simultaneous browser windows (one per timeframe)
- Real-time online learning model (PyTorch)
- Persistent replay buffer + running normalization
- Auto-saves model and labels after every batch

---

## Project Structure
