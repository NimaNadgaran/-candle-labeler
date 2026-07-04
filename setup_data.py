"""
setup_data.py
=================================================================
RUN THIS FILE EXACTLY ONCE, BEFORE THE FIRST TIME YOU RUN main.py.

It prepares the two data files the rest of the project depends on:

  1. data/xauusd_1min.csv
     A 1-minute candle file built from your raw tick/second-level
     price data. This is what chart_app.py reads and displays, and
     what model_agent.py extracts features from.
     Columns: candle_number,timestamp,open,high,low,close,volume

  2. my_data.csv
     An empty file (header row only) that will store every trading
     decision you make while labeling. main.py appends one row to
     this file after each batch of 6 windows is answered.
     Columns: random_number,f_300,f_200,f_100,f_50,f_20,f_10

You do NOT need to run this again unless:
  - You want to rebuild the candle file from a new/updated raw data file, or
  - my_data.csv gets deleted or corrupted and you need a fresh one
    (WARNING: create_csv_file() appends a header row every time it's
    called, so re-running this against an existing my_data.csv will
    add a second header row in the middle of your data - move or
    rename the old file first if you need to regenerate it).
=================================================================
"""

from output_file import convert_to_candles, create_csv_file

if __name__ == "__main__":
    # --- Step 1: build the 1-minute candle file used by the chart viewer ---
    # Put your own raw tick-data CSV under data/ first (see data/README.md
    # for the expected raw format). Update the input_file path below to
    # match whatever you name that raw file.
    convert_to_candles(
        input_file="data/xauusd_s1_20260529_20260628.csv",  # <-- your raw tick/second data
        output_file="data/xauusd_1min.csv",                  # <-- generated 1-min candles
        seconds=60,                                            # 60 = 1-minute candles
    )

    # --- Step 2: create the empty labeled-data file with the correct header ---
    create_csv_file("my_data.csv")

    print("\nSetup complete. You can now run: python main.py")
