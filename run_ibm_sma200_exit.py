"""Run Basis ADX SMA200 Exit on IBM 1D and save HTML report."""
import pandas as pd
import numpy as np
from pathlib import Path
from backtesting import Backtest
from strategies.basis_adx_sma200_exit import BasisAdxSma200Exit
import warnings; warnings.filterwarnings("ignore")

BASE = Path(r"C:\Users\satri\code\backtester")
REPORTS = BASE / "reports"
REPORTS.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(BASE / "IBM_1d.txt", parse_dates=["DateTime"])
df = df.rename(columns={"DateTime": "Date"}).set_index("Date").sort_index()
df = df.dropna(subset=["Open", "High", "Low", "Close"])
df = df[df["Volume"] > 0]

print(f"📊 IBM 1D — Basis ADX + SMA200 Exit")
print(f"   Baris: {len(df):,} ({df.index[0]} — {df.index[-1]})")

# Run backtest
bt = Backtest(df, BasisAdxSma200Exit, cash=100_000, commission=0.001, finalize_trades=True)
stats = bt.run()

# Summary
eq_final = stats['Equity Final [$]']
pnl = eq_final - 100_000
trades = stats['# Trades']

print(f"\n{'='*50}")
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
print(f"{'='*50}")

# Save report
report_name = "IBM_basis_adx_rr1_sma200exit_1d.html"
report_path = REPORTS / report_name
bt.plot(filename=str(report_path), open_browser=False)
print(f"\n📁 Report: {report_path}")
print("✅ SELESAI!")
