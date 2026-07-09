"""Quick test: Basis ADX Multi TF Reversal on IBM H1 (yfinance)."""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest
from strategies.basis_adx_multi_tf_reversal import BasisAdxMultiTfReversal

# ── Helpers ──
def calc_adx(c, h, l, period=14):
    tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
    up = np.diff(h, prepend=h[0]); dn = np.diff(l, prepend=l[0])
    pdm = np.where((up>dn)&(up>0), up, 0); mdm = np.where((dn>up)&(dn>0), dn, 0)
    atr = pd.Series(tr).rolling(period).mean().values
    sp = pd.Series(pdm).rolling(period).mean().values
    sm = pd.Series(mdm).rolling(period).mean().values
    pdi = np.where(atr>0, 100*sp/atr, 0); mdi = np.where(atr>0, 100*sm/atr, 0)
    dx = np.where((pdi+mdi)>0, 100*np.abs(pdi-mdi)/(pdi+mdi), 0)
    adx = pd.Series(dx).rolling(period).mean().values
    return adx, pdi, mdi

# ── Download data ──
print("Downloading IBM 1h...")
h1 = yf.download('IBM', period='2y', interval='1h', progress=False, auto_adjust=True)
if isinstance(h1.columns, pd.MultiIndex):
    h1.columns = h1.columns.get_level_values(0)
h1.columns = [c.lower() for c in h1.columns]
h1 = h1.rename(columns={
    'open': 'Open', 'high': 'High', 'low': 'Low',
    'close': 'Close', 'volume': 'Volume',
})
h1.index = pd.to_datetime(h1.index)
if h1.index.tz is not None:
    h1.index = h1.index.tz_localize(None)
print(f"  {len(h1)} bars | {h1.index[0]} → {h1.index[-1]}")

# ── Daily HTF ──
print("Downloading IBM Daily...")
daily = yf.download('IBM', period='5y', interval='1d', progress=False, auto_adjust=True)
if isinstance(daily.columns, pd.MultiIndex):
    daily.columns = daily.columns.get_level_values(0)
daily.columns = [c.lower() for c in daily.columns]
daily.index = pd.to_datetime(daily.index)
if daily.index.tz is not None:
    daily.index = daily.index.tz_localize(None)

dc, dh, dl = daily['close'].values, daily['high'].values, daily['low'].values
_, dp, dm = calc_adx(dc, dh, dl)
dsma20 = pd.Series(dc).rolling(20).mean().values

# Map daily to H1 bars by date
h1['date'] = h1.index.normalize()
daily_map = pd.DataFrame({'date': daily.index, 'htf_pdi': dp, 'htf_mdi': dm, 'htf_sma': dsma20}).set_index('date')
h1 = pd.merge_asof(h1.sort_values('date'), daily_map, left_on='date', right_index=True, direction='backward')
h1 = h1.dropna(subset=['htf_pdi', 'htf_mdi']).drop(columns=['date'])
print(f"  {len(h1)} bars with daily HTF data")

# ── Run backtest ──
bt = Backtest(h1, BasisAdxMultiTfReversal, cash=100_000, commission=0.001)
stats = bt.run()
print(f"\n{'='*50}")
print(f"  IBM H1 — Basis ADX Multi TF Reversal")
print(f"{'='*50}")
print(f"  Return    : {stats['Return [%]']:+.2f}%")
print(f"  Buy & Hold: {stats['Buy & Hold Return [%]']:+.2f}%")
print(f"  Trades    : {stats['# Trades']}")
print(f"  Win Rate  : {stats['Win Rate [%]']:.1f}%")
print(f"  Max DD    : {stats['Max. Drawdown [%]']:.2f}%")
print(f"  Sharpe    : {stats['Sharpe Ratio']:.2f}")
print(f"  Best/Worst: +{stats['Best Trade [%]']:.2f}% / {stats['Worst Trade [%]']:.2f}%")

# Show trade breakdown
trades = stats['_trades']
if trades is not None and len(trades) > 0:
    longs = trades[trades['Size'] > 0]
    shorts = trades[trades['Size'] < 0]
    print(f"\n  LONGs  : {len(longs)}  | Avg PnL: {longs['PnL'].mean():+.2f}" if len(longs) > 0 else "\n  LONGs  : 0")
    print(f"  SHORTs : {len(shorts)} | Avg PnL: {shorts['PnL'].mean():+.2f}" if len(shorts) > 0 else "  SHORTs : 0")
    print(f"\n  Last 5 trades:")
    print(trades[['Size', 'EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct']].tail(5).to_string())
else:
    print("  No trades")
