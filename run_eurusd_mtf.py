"""
Backtest Basis ADX Multi-TF pada EURUSD 1H + Daily DI confirmation.

Membandingkan:
  1) Standard Basis ADX (tanpa daily DI)
  2) Multi-TF Basis ADX (dengan daily +DI > +DI[5] untuk BUY,
     daily -DI > -DI[5] untuk SHORT)
"""
import warnings; warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from backtesting import Backtest
from strategies.basis_adx_strategy import BasisAdxStrategy
from strategies.bb_adx_strategy import add_daily_di

BASE = Path(r"C:\Users\satri\code\backtester")

# ── Load EURUSD 1H data ───────────────────────────────────
print("📂 Loading EURUSD 1H data...")
df = pd.read_csv(BASE / "EURUSD_1h_yf.txt")
# Parse datetime with UTC, then remove timezone
df["Date"] = pd.to_datetime(df["Datetime"], utc=True)
df = df.drop(columns=["Datetime"])
df = df.set_index("Date").sort_index()
df.index = df.index.tz_localize(None)  # remove tz for backtesting.py
df = df.dropna(subset=["Open", "High", "Low", "Close"])
keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
df = df[keep]
print(f"   {len(df)} bars ({df.index[0]} — {df.index[-1]})")

# ── Run 1: Standard (no daily DI) ─────────────────────────
print("\n" + "=" * 55)
print("  RUN 1: STANDARD Basis ADX (tanpa daily DI)")
print("=" * 55)

bt1 = Backtest(df, BasisAdxStrategy, cash=100_000, commission=0.0001, finalize_trades=True)
s1 = bt1.run()

print(f"  Return       : {s1['Return [%]']:.2f}%")
print(f"  Buy & Hold   : {s1['Buy & Hold Return [%]']:.2f}%")
print(f"  Sharpe       : {s1['Sharpe Ratio']:.2f}")
print(f"  Max DD       : {s1['Max. Drawdown [%]']:.2f}%")
print(f"  Trades       : {s1['# Trades']}")
print(f"  Win Rate     : {s1.get('Win Rate [%]', 0):.1f}%")
print(f"  Best Trade   : {s1.get('Best Trade [%]', 0):.2f}%")
print(f"  Worst Trade  : {s1.get('Worst Trade [%]', 0):.2f}%")

# ── Run 2: Multi-TF (dengan daily DI) ─────────────────────
print("\n" + "=" * 55)
print("  RUN 2: MULTI-TF Basis ADX (+ daily DI confirmation)")
print("=" * 55)

print("   Menambahkan daily DI columns...")
df_mtf = add_daily_di(df.copy())

# Cek berapa bar yang punya daily DI valid
valid_daily = df_mtf['daily_pdi'].notna().sum()
print(f"   {valid_daily} bar dengan daily PDI valid (dari {len(df_mtf)})")

bt2 = Backtest(df_mtf, BasisAdxStrategy, cash=100_000, commission=0.0001, finalize_trades=True)
s2 = bt2.run()

trades2 = s2.get('_trades', None)
if trades2 is not None and len(trades2) > 0:
    longs = trades2[trades2['Size'] > 0]
    shorts = trades2[trades2['Size'] < 0]
    print(f"   Long trades : {len(longs)}")
    print(f"   Short trades: {len(shorts)}")
    if len(shorts) > 0:
        print(f"   Short win % : {shorts['ReturnPct'].gt(0).mean()*100:.1f}%")

print(f"  Return       : {s2['Return [%]']:.2f}%")
print(f"  Buy & Hold   : {s2['Buy & Hold Return [%]']:.2f}%")
print(f"  Sharpe       : {s2['Sharpe Ratio']:.2f}")
print(f"  Max DD       : {s2['Max. Drawdown [%]']:.2f}%")
print(f"  Trades       : {s2['# Trades']}")
print(f"  Win Rate     : {s2.get('Win Rate [%]', 0):.1f}%")
print(f"  Best Trade   : {s2.get('Best Trade [%]', 0):.2f}%")
print(f"  Worst Trade  : {s2.get('Worst Trade [%]', 0):.2f}%")

# ── Summary ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  📊 PERBANDINGAN")
print("=" * 55)
print(f"{'Metric':<20} {'Standard':>12} {'Multi-TF':>12}")
print(f"{'-'*20} {'-'*12} {'-'*12}")
for metric, attr in [
    ("Return %", "Return [%]"),
    ("Sharpe", "Sharpe Ratio"),
    ("Max DD %", "Max. Drawdown [%]"),
    ("Trades", "# Trades"),
    ("Win Rate %", "Win Rate [%]"),
]:
    v1 = s1.get(attr, 0)
    v2 = s2.get(attr, 0)
    if isinstance(v1, float):
        print(f"{metric:<20} {v1:>12.2f} {v2:>12.2f}")
    else:
        print(f"{metric:<20} {str(v1):>12} {str(v2):>12}")

# Save reports
bt1.plot(filename=str(BASE / "reports" / "EURUSD_basis_adx_1h_standard.html"), open_browser=False)
bt2.plot(filename=str(BASE / "reports" / "EURUSD_basis_adx_1h_multitf.html"), open_browser=False)
print(f"\n📁 Report standard: reports/EURUSD_basis_adx_1h_standard.html")
print(f"📁 Report multi-TF: reports/EURUSD_basis_adx_1h_multitf.html")
print("\n✅ Selesai!")
