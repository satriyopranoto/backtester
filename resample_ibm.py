"""
Resample IBM 1-minute data into multiple timeframes.
Input: IBM.txt (MM/DD/YYYY,HH:MM,Open,High,Low,Close,Volume)
Output: IBM_5min.txt, IBM_10min.txt, IBM_30min.txt, IBM_1h.txt, IBM_2h.txt, IBM_4h.txt, IBM_1d.txt
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

INPUT = r"C:\Users\satri\code\backtester\IBM.txt"
OUTDIR = r"C:\Users\satri\code\backtester"
CHUNKSIZE = 500_000  # process 500k rows at a time

# Define columns
# Define columns
COL_NAMES = ["DateStr", "TimeStr", "Open", "High", "Low", "Close", "Volume"]
COL_TYPES = {
    "DateStr": "object", "TimeStr": "object",
    "Open": "float64", "High": "float64",
    "Low": "float64", "Close": "float64", "Volume": "int64"
}

# Timeframes to generate
TIMEFRAMES = {
    "5min": "5min",
    "10min": "10min",
    "30min": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1D",
}

# Open output files with headers
outfiles = {}
for tf_name in TIMEFRAMES:
    path = os.path.join(OUTDIR, f"IBM_{tf_name}.txt")
    outfiles[tf_name] = open(path, "w")
    outfiles[tf_name].write("DateTime,Open,High,Low,Close,Volume\n")

total_rows = 0
chunk_num = 0

print("Processing IBM.txt in chunks...")
start_time = datetime.now()

for chunk in pd.read_csv(
    INPUT,
    header=None,
    names=COL_NAMES,
    dtype=COL_TYPES,
    chunksize=CHUNKSIZE,
    low_memory=True,
):
    chunk_num += 1
    total_rows += len(chunk)
    
    # Combine Date+Time into single datetime - file col 0 = MM/DD/YYYY, col 1 = HH:MM
    chunk["DateTime"] = pd.to_datetime(
        chunk.iloc[:, 0].astype(str) + " " + chunk.iloc[:, 1].astype(str),
        format="%m/%d/%Y %H:%M",
        errors="coerce"
    )
    # Drop the raw Date and Time columns, keep only the named ones
    chunk = chunk[["DateTime", "Open", "High", "Low", "Close", "Volume"]]
    
    # Drop any NaT
    before = len(chunk)
    chunk = chunk.dropna(subset=["DateTime"])
    if len(chunk) < before:
        print(f"  Chunk {chunk_num}: dropped {before - len(chunk)} unparseable rows")
    
    # Set datetime as index for resampling
    chunk = chunk.set_index("DateTime")
    
    # Resample for each timeframe
    for tf_name, tf_rule in TIMEFRAMES.items():
        resampled = chunk.resample(tf_rule).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        # Drop rows where all OHLC are NaN (no data in that period)
        resampled = resampled.dropna(subset=["Open", "High", "Low", "Close"], how="all")
        
        if len(resampled) > 0:
            # Write to file
            resampled.to_csv(outfiles[tf_name], header=False, date_format="%m/%d/%Y %H:%M")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"  Chunk {chunk_num}: {len(chunk):,} rows processed ({total_rows:,} total, {elapsed:.0f}s)")

# Close all files
for f in outfiles.values():
    f.close()

# Count results
print("\n=== Results ===")
for tf_name in TIMEFRAMES:
    path = os.path.join(OUTDIR, f"IBM_{tf_name}.txt")
    with open(path) as f:
        line_count = sum(1 for _ in f) - 1  # subtract header
    size_mb = os.path.getsize(path) / 1_000_000
    print(f"  IBM_{tf_name}.txt: {line_count:,} rows, {size_mb:.1f} MB")

elapsed_total = (datetime.now() - start_time).total_seconds()
print(f"\nTotal time: {elapsed_total:.0f}s")
print(f"Total input rows: {total_rows:,}")
