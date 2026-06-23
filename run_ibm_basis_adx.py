"""
Run Basis ADX backtest on IBM daily data from local file.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from backtesting import Backtest
from strategies.basis_adx_strategy import BasisAdxStrategy

# ── Load IBM 1D data ──────────────────────────────────────────
DATA_FILE = r"C:\Users\satri\code\backtester\IBM_1d.txt"

print("📂 Loading IBM 1D data...")
df = pd.read_csv(DATA_FILE, parse_dates=["DateTime"])
df = df.rename(columns={"DateTime": "Date"})
df = df.set_index("Date")
# Sort ascending
df = df.sort_index()

print(f"   {len(df)} baris ({df.index[0].date()} — {df.index[-1].date()})")
print(f"   Range: ${df['Low'].min():.2f} — ${df['High'].max():.2f}")
print()

# ── Run backtest ──────────────────────────────────────────────
CASH = 100_000  # $100k

bt = Backtest(
    df,
    BasisAdxStrategy,
    cash=CASH,
    commission=0.001,
    finalize_trades=True,
)

print("🚀 Running Basis ADX backtest...")
stats = bt.run()

# ── Print results ─────────────────────────────────────────────
eq_final = stats['Equity Final [$]']
pnl = eq_final - CASH

print(f"\n{'='*55}")
print(f"  📊 HASIL BACKTEST — IBM 1D")
print(f"  Strategy: Basis + ADX")
print(f"{'='*55}")
print(f"  Periode      : {stats['Start'].date()} — {stats['End'].date()}")
print(f"  Return       : {stats['Return [%]']:.2f}%")
print(f"  Buy & Hold   : {stats['Buy & Hold Return [%]']:.2f}%")
print(f"  PnL Total    : $ {pnl:,.0f}")
print(f"  Equity Final : $ {eq_final:,.0f}")
print(f"  Equity Peak  : $ {stats['Equity Peak [$]']:,.0f}")
print(f"  Max DD       : {stats['Max. Drawdown [%]']:.2f}%")
print(f"  Sharpe Ratio : {stats['Sharpe Ratio']:.2f}")
print(f"  Trades       : {stats['# Trades']}")
print(f"  Win Rate     : {stats.get('Win Rate [%]', float('nan')):.1f}%")
print(f"  Best Trade   : {stats.get('Best Trade [%]', float('nan')):.2f}%")
print(f"  Worst Trade  : {stats.get('Worst Trade [%]', float('nan')):.2f}%")
print(f"  Avg Trade    : {stats.get('Avg Trade [%]', float('nan')):.2f}%")

trades = stats.get('_trades', None)
if trades is not None and len(trades) > 0:
    print(f"  ──────────────────────────────────────")
    print(f"  Last 8 trades:")
    for i, t in trades.tail(8).iterrows():
        ret = t.ReturnPct * 100
        emoji = "🔥" if ret >= 0 else "❌"
        label = " cut loss" if ret < 0 else ""
        print(f"  #{i}: {t.EntryTime.date()} → {t.ExitTime.date()}  "
              f"| PnL: {ret:+.2f}%  ($ {t.PnL:+,.0f}) {emoji}{label}")
else:
    print(f"  ⚠️  No trades executed.")

print(f"{'='*55}\n")
print("💡 Gunakan --save untuk simpan laporan HTML")
print("✅ Selesai!")
