"""
Backtest 3 variant Basis ADX pada GC=F (Gold Futures) 1H.

Membandingkan:
  1) STANDARD     — H1 only (tanpa daily DI)
  2) MULTI-TF     — STANDARD + daily +DI > daily -DI (LONG) / daily -DI > daily +DI (SHORT)
  3) MULTI-TF MODIF — MULTI-TF + daily +DI > daily +DI[1] (LONG) / daily -DI > daily -DI[1] (SHORT)
"""
import warnings; warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from backtesting import Backtest
from strategies.basis_adx_strategy import (
    BasisAdxStrategy,
    BasisAdxMultiTf,
    BasisAdxMultiTfModif,
)
from strategies.bb_adx_strategy import add_daily_di

BASE = Path(r"C:\Users\satri\code\backtester")

# ── Load GC=F 1H data ──────────────────────────────────────
print("📂 Loading GC=F 1H data...")
df = pd.read_csv(BASE / "GC_F_1h_yf.txt")
df["Date"] = pd.to_datetime(df["Datetime"], utc=True)
df = df.drop(columns=["Datetime"])
df = df.set_index("Date").sort_index()
df.index = df.index.tz_localize(None)
df = df.dropna(subset=["Open", "High", "Low", "Close"])
keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
df = df[keep]
print(f"   {len(df)} bars ({df.index[0]} — {df.index[-1]})")
print(f"   Range: ${df['Low'].min():.2f} — ${df['High'].max():.2f}")

# ── Add daily DI columns for multi-TF variants ──
print("\n📊 Menambahkan daily DI columns...")
df_mtf = add_daily_di(df.copy())
valid_daily = df_mtf['daily_pdi'].notna().sum()
print(f"   {valid_daily} bar dengan daily PDI valid (dari {len(df_mtf)})")

# ── Run all 3 ──
runs = [
    ("STANDARD", BasisAdxStrategy, df, "tanpa daily DI"),
    ("MULTI-TF", BasisAdxMultiTf, df_mtf, "daily +DI > -DI / -DI > +DI"),
    ("MODIF",    BasisAdxMultiTfModif, df_mtf, "daily +DI > -DI + rising(1)"),
]

results = {}
for name, strat, data, desc in runs:
    print(f"\n{'='*55}")
    print(f"  {name}: {desc}")
    print(f"{'='*55}")
    bt = Backtest(data, strat, cash=100_000, commission=0.001, finalize_trades=True)
    s = bt.run()
    results[name] = (s, bt)

    trades = s.get('_trades', None)
    n_long = len(trades[trades['Size'] > 0]) if trades is not None and len(trades) > 0 else 0
    n_short = len(trades[trades['Size'] < 0]) if trades is not None and len(trades) > 0 else 0

    print(f"  Return       : {s['Return [%]']:.2f}%")
    print(f"  Buy & Hold   : {s['Buy & Hold Return [%]']:.2f}%")
    print(f"  Sharpe       : {s['Sharpe Ratio']:.2f}")
    print(f"  Max DD       : {s['Max. Drawdown [%]']:.2f}%")
    print(f"  Trades       : {s['# Trades']} ({n_long}L / {n_short}S)")
    print(f"  Win Rate     : {s.get('Win Rate [%]', 0):.1f}%")
    print(f"  Best Trade   : {s.get('Best Trade [%]', 0):.2f}%")
    print(f"  Worst Trade  : {s.get('Worst Trade [%]', 0):.2f}%")

# ── Summary ──
print(f"\n\n{'='*60}")
print(f"  📊 PERBANDINGAN — GC=F 1H")
print(f"{'='*60}")
headers = ["Metric", "STANDARD", "MULTI-TF", "MODIF"]
print(f"  {headers[0]:<20} {headers[1]:>12} {headers[2]:>12} {headers[3]:>12}")
print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
for metric, attr in [
    ("Return %", "Return [%]"),
    ("Buy & Hold", "Buy & Hold Return [%]"),
    ("Sharpe", "Sharpe Ratio"),
    ("Max DD %", "Max. Drawdown [%]"),
    ("Trades", "# Trades"),
    ("Win Rate %", "Win Rate [%]"),
    ("Best Trade %", "Best Trade [%]"),
    ("Worst Trade %", "Worst Trade [%]"),
]:
    vals = []
    for name, _, _, _ in runs:
        s, _ = results[name]
        v = s.get(attr, 0)
        vals.append(v)
    if isinstance(vals[0], float):
        print(f"  {metric:<20} {vals[0]:>12.2f} {vals[1]:>12.2f} {vals[2]:>12.2f}")
    else:
        print(f"  {metric:<20} {str(vals[0]):>12} {str(vals[1]):>12} {str(vals[2]):>12}")

# ── Save HTML reports ──
reports_dir = BASE / "reports"
reports_dir.mkdir(exist_ok=True)
for name, _, _, desc in runs:
    _, bt = results[name]
    fname = reports_dir / f"GCF_basis_adx_1h_{name.lower()}.html"
    bt.plot(filename=str(fname), open_browser=False)
    print(f"  📁 {name}: {fname.name}")

print("\n✅ Selesai!")
