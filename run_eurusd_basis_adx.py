"""Backtest Basis ADX on EURUSD from yfinance data.
Compares 30min, 1H, and 2H timeframes.
"""
import pandas as pd, numpy as np
from pathlib import Path
from backtesting import Backtest
from strategies.basis_adx_strategy import BasisAdxStrategy
import warnings; warnings.filterwarnings("ignore")

BASE = Path(r"C:\Users\satri\code\backtester")

def load_yf_data(filepath):
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.set_index("Date").sort_index()
    # Convert to timezone-naive for backtest compatibility
    df.index = df.index.tz_localize(None)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    # Keep only OHLCV columns (drop Dividends, Stock Splits)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep]
    return df

# ── Run on 1H (1 year data) ──
print("=" * 55)
print("  EURUSD — 1H (1 year)")
print("=" * 55)
df1h = load_yf_data(BASE / "EURUSD_1h_yf.txt")
print(f"  Bars: {len(df1h)} ({df1h.index[0]} — {df1h.index[-1]})")
bt1h = Backtest(df1h, BasisAdxStrategy, cash=100_000, commission=0.0001, finalize_trades=True)
s1h = bt1h.run()
print(f"  Return : {s1h['Return [%]']:.2f}%")
print(f"  Sharpe : {s1h['Sharpe Ratio']:.2f}")
print(f"  Max DD : {s1h['Max. Drawdown [%]']:.2f}%")
print(f"  Trades : {s1h['# Trades']}")
print(f"  Win%   : {s1h.get('Win Rate [%]', 0):.1f}%")
print(f"  Best   : {s1h.get('Best Trade [%]', 0):.2f}%")
print(f"  Worst  : {s1h.get('Worst Trade [%]', 0):.2f}%")
bt1h.plot(filename=str(BASE/"reports"/"EURUSD_basis_adx_1h.html"), open_browser=False)
print(f"  Report : reports/EURUSD_basis_adx_1h.html")

# ── 2H via resample from 1H ──
print("\n" + "=" * 55)
print("  EURUSD — 2H (resampled)")
print("=" * 55)
df2h = df1h.resample("2h").agg({
    "Open": "first", "High": "max", "Low": "min",
    "Close": "last", "Volume": "sum"
}).dropna()
print(f"  Bars: {len(df2h)} ({df2h.index[0]} — {df2h.index[-1]})")
bt2h = Backtest(df2h, BasisAdxStrategy, cash=100_000, commission=0.0001, finalize_trades=True)
s2h = bt2h.run()
print(f"  Return : {s2h['Return [%]']:.2f}%")
print(f"  Sharpe : {s2h['Sharpe Ratio']:.2f}")
print(f"  Max DD : {s2h['Max. Drawdown [%]']:.2f}%")
print(f"  Trades : {s2h['# Trades']}")
print(f"  Win%   : {s2h.get('Win Rate [%]', 0):.1f}%")
print(f"  Best   : {s2h.get('Best Trade [%]', 0):.2f}%")
print(f"  Worst  : {s2h.get('Worst Trade [%]', 0):.2f}%")
bt2h.plot(filename=str(BASE/"reports"/"EURUSD_basis_adx_2h.html"), open_browser=False)
print(f"  Report : reports/EURUSD_basis_adx_2h.html")

# ── Summary ──
print("\n" + "=" * 55)
print("  SUMMARY")
print("=" * 55)
print(f"  {'1H':6s} | Return: {s1h['Return [%]']:>7.2f}% | Sharpe: {s1h['Sharpe Ratio']:.2f} | DD: {s1h['Max. Drawdown [%]']:.2f}% | Trades: {s1h['# Trades']} | Win%: {s1h.get('Win Rate [%]', 0):.1f}% | Best: {s1h.get('Best Trade [%]', 0):.2f}% | Worst: {s1h.get('Worst Trade [%]', 0):.2f}%")
print("\n✅ DONE")
