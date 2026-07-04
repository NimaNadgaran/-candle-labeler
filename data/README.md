# Data Folder Guide

This folder contains the data used by the **Candle Labeler** project.

---

## 1. Raw Tick Data (Input)

**File example:** `xauusd_s1_20260529_20260628.csv`

This is the original high-frequency data you downloaded.

**Source:** [marketdata-hub.com](https://marketdata-hub.com)

**Format (First 6 rows):**

```csv
timestamp,open,high,low,close,volume
1780012800000,4494.795,4495.145,4494.535,4494.535,0.0014
1780012801000,4494.565,4494.565,4493.635,4493.705,0.0012
1780012802000,4493.335,4493.725,4493.335,4493.678,0.0033
1780012803000,4493.698,4494.348,4493.698,4493.948,0.0022
1780012804000,4493.948,4494.125,4493.775,4493.815,0.002
1780012805000,4493.815,4493.855,4492.888,4492.888,0.0015
```
Columns:

- `timestamp`: Unix timestamp in milliseconds
- open, high, low, close: Price levels
- volume: Trading volume for that second


## 2. Converted 1-Minute Candles (Generated)
File: `xauusd_1min.csv`
This file is automatically created when you run `setup_data.py`.
Format (First 6 rows):
```csv
csvcandle_number,timestamp,open,high,low,close,volume
1,1780012800000,4494.795,4495.145,4492.655,4494.205,0.0587
2,1780012860000,4494.185,4497.895,4494.175,4495.885,0.0498
3,1780012920000,4495.845,4496.175,4493.778,4493.778,0.0303
4,1780012980000,4493.478,4496.085,4492.125,4492.568,0.0639
5,1780013040000,4492.568,4495.355,4492.198,4494.635,0.0545
6,1780013100000,4494.605,4496.205,4493.875,4495.335,0.0499
```
Columns:

`candle_number`: Sequential ID of the candle
`timestamp`: Start time of the 1-minute candle (milliseconds)
`open`, `high`, `low`, `close`: Standard OHLC
`volume`: Total volume during that 1-minute period

*This is the file actually used by `main.py1 and 1chart_app.py`.*

##3. Labeled Data (`my_data.csv`)
This file stores all your manual labeling decisions.
**Example Head:**
```csv
csvrandom_number,f_300,f_200,f_100,f_50,f_20,f_10
12345,buy,neutral,sell,neutral,buy,neutral
67890,neutral,buy,buy,sell,neutral,neutral
```
Columns:

`random_number`: The candle number you labeled
`f_300`, `f_200`, `f_100`, `f_50`, `f_20`, `f_10`: Your decision for each timeframe (`buy`, `sell`, or `neutral`)

This file grows as you label more batches.

Important Notes

Never delete `xauusd_1min.csv` unless you want to rebuild it using `setup_data.py`.
`my_data.csv` is your training data — keep it safe.
You can add more raw data files in the future and re-run `setup_data.py` (just update the filename inside the script).
