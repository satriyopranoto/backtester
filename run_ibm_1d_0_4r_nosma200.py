"""Run Basis ADX 1D — 0.4R TP, No SMA200 + save HTML report."""
import pandas as pd, numpy as np
from pathlib import Path
from backtesting import Backtest
from strategies.basis_adx_strategy import BasisAdxStrategy
import warnings; warnings.filterwarnings("ignore")

BASE = Path(r"C:\Users\satri\code\backtester")
df = pd.read_csv(BASE/"IBM_1d.txt", parse_dates=["DateTime"])
df = df.rename(columns={"DateTime":"Date"}).set_index("Date").sort_index()
df = df.dropna(subset=["Open","High","Low","Close"])
df = df[df["Volume"]>0]

bt = Backtest(df, BasisAdxStrategy, cash=100_000, commission=0.001, finalize_trades=True)
stats = bt.run()

print(f"\n{'='*60}")
print(f"  IBM 1D — Basis ADX (0.4R TP) — NO SMA 200")
print(f"{'='*60}")
print(f"  Return       : {stats['Return [%]']:.2f}%")
print(f"  Buy & Hold   : {stats['Buy & Hold Return [%]']:.2f}%")
print(f"  Equity Final : ${stats['Equity Final [$]']:,.2f}")
print(f"  Sharpe       : {stats['Sharpe Ratio']:.2f}")
print(f"  Max DD       : {stats['Max. Drawdown [%]']:.2f}%")
print(f"  # Trades     : {stats['# Trades']}")
print(f"  Win Rate     : {stats.get('Win Rate [%]', 0):.1f}%")
print(f"  Best Trade   : {stats.get('Best Trade [%]', 0):.2f}%")
print(f"  Worst Trade  : {stats.get('Worst Trade [%]', 0):.2f}%")
print(f"  Avg Trade    : {stats.get('Avg Trade [%]', 0):.2f}%")

report_path = BASE/"reports"/"IBM_basis_adx_0_4r_nosma200_1d.html"
bt.plot(filename=str(report_path), open_browser=False)
print(f"\n  Report       : {report_path}")
print("✅ DONE")
