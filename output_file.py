"""
output_file.py
=================================================================
Data-preparation utilities for the candle-labeler project.

This file is not run directly during normal use - its two functions
are called once by setup_data.py before you ever start main.py:

  convert_to_candles()  -> builds the 1-minute OHLCV candle file from
                           raw tick/second-level price data.
  create_csv_file()     -> creates the empty my_data.csv file (with
                           the correct header) that main.py appends
                           your labeled decisions to.

add_row() is a small standalone helper (not used by the main app
loop - main.py writes rows itself via pandas) kept here for manual/
ad-hoc use if you ever want to append a row to a CSV from a script
or the REPL without going through pandas.
=================================================================
"""

import csv
import os
import pandas as pd


def convert_to_candles(input_file, output_file, seconds):
    """
    Convert raw tick/second-level price data into fixed-interval OHLCV
    candles and write the result to output_file with a header row.

    Args:
        input_file (str):  path to the raw data CSV. Expected to have
                            either 6 columns (timestamp, open, high,
                            low, close, volume) or 7 columns
                            (timestamp, symbol_id, open, high, low,
                            close, volume) - a header row is optional,
                            it's auto-detected and stripped if present.
        output_file (str): path to write the resulting candle CSV to.
        seconds (int):      candle interval in seconds (60 = 1-minute
                            candles, as used by the rest of this project).

    Returns:
        pandas.DataFrame: the candle data that was also written to disk,
        with columns: candle_number, timestamp, open, high, low,
        close, volume.
    """

    # ---------- 1. Read file, detect and remove header ----------
    df = pd.read_csv(input_file, header=None, dtype=str, low_memory=False)

    # Check if first row looks like a header (contains text)
    first_row = df.iloc[0].astype(str).str.lower()
    header_keywords = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'index']
    if any(first_row.str.contains('|'.join(header_keywords), case=False, na=False)):
        df = df.iloc[1:].reset_index(drop=True)   # remove header row

    # ---------- 2. Assign column names based on number of columns ----------
    num_cols = df.shape[1]
    if num_cols == 6:
        # Columns: timestamp, open, high, low, close, volume
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    elif num_cols == 7:
        # Columns: timestamp, symbol_id, open, high, low, close, volume
        df.columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        df = df.drop(columns=['symbol'])   # we don't need symbol ID
    else:
        raise ValueError(f"Unexpected number of columns: {num_cols}. Expected 6 or 7.")

    # ---------- 3. Convert OHLCV columns to numeric ----------
    # These are read in as strings above (dtype=str), so they must be
    # converted to numeric here - otherwise max/min/sum on 'high'/'low'/
    # 'volume' would silently do lexicographic string comparisons /
    # string concatenation instead of numeric math.
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where price/volume conversion failed (corrupt rows)
    before = len(df)
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume']).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Warning: dropped {dropped} row(s) with non-numeric OHLCV values.")

    # ---------- 4. Convert timestamp to milliseconds ----------
    # Don't gate this on `dtype == 'object'` - on pandas >=2.x with the
    # "future string" option (and pandas 3.0 by default), string columns
    # get dtype 'str', not 'object', so that check would silently skip
    # conversion and leave timestamp as text, crashing the numeric
    # comparison further down. is_numeric_dtype() is dtype-agnostic.
    if not pd.api.types.is_numeric_dtype(df['timestamp']):
        ts_numeric = pd.to_numeric(df['timestamp'], errors='coerce')
        if ts_numeric.notna().all():
            df['timestamp'] = ts_numeric
        else:
            # If it's a datetime string, convert to milliseconds
            df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6

    # If timestamp values are in seconds (not milliseconds), convert to
    # milliseconds. The threshold here is 1e11: epoch-seconds for any
    # real-world date already exceeds 1e9 (so a naive "< 1e9" check would
    # misidentify real seconds-resolution data as "already milliseconds"
    # and skip the conversion, producing dates around 1970). Epoch-ms for
    # current dates is around 1.7e12, and epoch-ms won't reach 1e11 until
    # the year 1973 - so 1e11 safely separates "seconds" from "milliseconds"
    # for any realistic trading data.
    if df['timestamp'].max() < 100_000_000_000:
        df['timestamp'] = df['timestamp'] * 1000   # seconds -> milliseconds

    # Create datetime column for grouping
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    # ---------- 5. Aggregate into candles ----------
    # Candle boundaries are aligned to absolute clock time (epoch-based)
    # via dt.floor(), not relative to the first row's timestamp - relative
    # alignment would make boundaries shift depending on what the first
    # tick in the file happens to be, which breaks consistency across
    # different files/chunks of the same instrument.
    df['candle_start'] = df['datetime'].dt.floor(f'{seconds}s')

    candles = df.groupby('candle_start').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()

    candles['candle_number'] = range(1, len(candles) + 1)

    # astype('int64') on a datetime column assumes nanosecond resolution
    # to land on milliseconds after //10**6. Pandas 3.0 defaults
    # datetime64 columns to microsecond resolution, which would silently
    # produce truncated/garbage timestamps with that approach. Dividing
    # by a Timedelta instead works regardless of the underlying
    # datetime64 unit.
    candles['timestamp'] = (candles['candle_start'] - pd.Timestamp('1970-01-01')) // pd.Timedelta(milliseconds=1)
    candles = candles[['candle_number', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # ---------- 6. Save with header ----------
    candles.to_csv(output_file, index=False, header=True)

    return candles


def create_csv_file(filename):
    """
    Create (or append a header row to) the CSV file that stores every
    labeling decision made in main.py.

    Args:
        filename (str): path of the CSV file to create, e.g. "my_data.csv".

    Note:
        This opens the file in append mode ('a'), so calling it more than
        once against an existing file will add a second header row rather
        than overwriting - only call this once per fresh file (see the
        warning in setup_data.py).
    """
    headers = ['random_number', 'f_300', 'f_200', 'f_100',
               'f_50', 'f_20', 'f_10']

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    print(f"CSV file '{filename}' created successfully with headers.")


def add_row(random_number, f300, f200, f100, f50, f20, f10, filename="my_data.csv"):
    """
    Append a single row to a labeled-data CSV file.

    This is a standalone convenience helper - the main application loop
    in main.py does NOT call this; it appends rows itself via pandas
    (df_new.to_csv(..., mode='a')). Use this only for manual/ad-hoc
    edits from a script or the Python REPL.

    Args:
        random_number (int): the candle number this row corresponds to.
        f300, f200, f100, f50, f20, f10: the labeled decision
            ("buy" / "sell" / "neutral") for each timeframe lookback.
        filename (str): the CSV file to append to (default "my_data.csv").
    """
    # Check if file exists, if not create it
    if not os.path.exists(filename):
        print(f"File '{filename}' not found.")
        return

    row_data = [random_number, f300, f200, f100, f50, f20, f10]

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_data)

    print(f"Row added successfully to '{filename}': {row_data}")


# Example usage (manual testing only - not part of the app's normal flow):
if __name__ == "__main__":
    # Create a CSV file
    create_csv_file("my_data.csv")

    # Add some rows
    add_row(42, True, False, True, False, True, False)
    add_row(73, 1, 0, 1, 0, 1, 0)
    add_row(99, "Yes", "No", "Yes", "No", "Yes", "No")
