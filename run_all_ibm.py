"""
Run Basis ADX backtest on ALL IBM timeframes sequentially.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from backtesting import Backtest
from strategies.basis_adx_strategy import BasisAdxStrategy
import time

CASH = 100_000  # $100k

FILES = [
    ("IBM_4h.txt",   "4 jam"),
    ("IBM_2h.txt",   "2 jam"),
    ("IBM_1h.txt",   "1 jam"),
    ("IBM_30min.txt","30 menit"),
    ("IBM_10min.txt","10 menit"),
    ("IBM_5min.txt", "5 menit"),
]

BASE = Path(r"C:\Users\satri\code\backtester")
REPORTS = BASE / "reports"
REPORTS.mkdir(exist_ok=True)

for fname, label in FILES:
    path = BASE / fname
    print(f"\n{'='*60}")
    print(f"📊 IBM — {label} ({fname})")
    print(f"{'='*60}")
    
    t0 = time.time()
    
    # Load
    df = pd.read_csv(path, parse_dates=["DateTime"])
    df = df.rename(columns={"DateTime": "Date"})
    df = df.set_index("Date").sort_index()
    
    # Drop rows where all OHLC are 0 or NaN
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[df["Volume"] > 0]
    
    print(f"   {len(df):,} baris ({df.index[0]} — {df.index[-1]})")
    
    # Run backtest
    bt = Backtest(df, BasisAdxStrategy, cash=CASH, commission=0.001, finalize_trades=True)
    stats = bt.run()
    
    # Print summary
    eq_final = stats['Equity Final [$]']
    pnl = eq_final - CASH
    trades = stats['# Trades']
    
    print(f"   Return       : {stats['Return [%]']:.2f}%")
    print(f"   Buy & Hold   : {stats['Buy & Hold Return [%]']:.2f}%")
    print(f"   PnL          : $ {pnl:+,.0f}")
    print(f"   Equity Final : $ {eq_final:,.0f}")
    print(f"   Sharpe       : {stats['Sharpe Ratio']:.2f}")
    print(f"   Max DD       : {stats['Max. Drawdown [%]']:.2f}%")
    print(f"   Trades       : {trades}")
    if trades > 0:
        print(f"   Win Rate     : {stats.get('Win Rate [%]', 0):.1f}%")
        print(f"   Best/Worst   : {stats.get('Best Trade [%]', 0):.2f}% / {stats.get('Worst Trade [%]', 0):.2f}%")
    
    # Save HTML report
    report_name = f"IBM_basis_adx_{fname.replace('.txt','')}.html"
    bt.plot(filename=str(REPORTS / report_name), open_browser=False)
    
    elapsed = time.time() - t0
    print(f"   ⏱ {elapsed:.0f}s  |  📁 reports/{report_name}")

print(f"\n{'='*60}")
print("✅ SEMUA SELESAI!")
print(f"{'='*60}")
