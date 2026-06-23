#!/usr/bin/env python3
"""Check BTPS.JK ADX behavior"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import yfinance as yf

df = yf.download('BTPS.JK', period='6mo', progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

close = df['Close'].astype(float)
high = df['High'].astype(float)
low = df['Low'].astype(float)
basis = close.rolling(20).mean()

period = 14
prev_close = close.shift(1)
tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
up_move = high - high.shift(1)
down_move = low.shift(1) - low
plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index)
minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index)
alpha = 1.0 / period
smoothed_tr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
smoothed_plus = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
smoothed_minus = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
pdi = 100 * smoothed_plus / smoothed_tr.replace(0, np.nan)
mdi = 100 * smoothed_minus / smoothed_tr.replace(0, np.nan)
dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

ero = int(2.8 * 10)
r_prev = high.rolling(ero).max().shift(1)
s_prev = low.rolling(ero).min().shift(1)
r_curr = high.rolling(ero).max()
s_curr = low.rolling(ero).min()
ab = np.where(high > r_prev, 1, np.where(low < s_prev, -1, 0))
ac = pd.Series(ab).replace(0, np.nan).ffill().fillna(0)
sl = pd.Series(np.where(ac == 1, s_curr, r_curr), index=df.index)

print(f"{'Date':<12} {'Close':>8} {'Basis':>8} {'SL':>8} {'ADX':>6} {'ADXchg':>7} {'PDI':>6} {'MDI':>6} {'PDI>MDI':>8} {'ADX>25':>7} {'Cond':>6}")
print("="*85)
for i in range(len(df)-30, len(df)):
    adx_rising = adx.iloc[i] > (adx.iloc[i-5] if i >= 5 else 0)
    adx_chg = adx.iloc[i] - (adx.iloc[i-5] if i >= 5 else 0)
    pdi_rising = pdi.iloc[i] > (pdi.iloc[i-5] if i >= 5 else 0)
    cond_old = (low.iloc[i] > sl.iloc[i] and close.iloc[i] > basis.iloc[i] and adx.iloc[i] > 25 and pdi.iloc[i] > mdi.iloc[i] and pdi_rising)
    cond_new = cond_old and adx_rising
    label = ''
    if cond_new: label = 'NEW✓'
    elif cond_old: label = 'OLD!'
    print(f"{df.index[i].strftime('%Y-%m-%d'):<12} {close.iloc[i]:>8.0f} {basis.iloc[i]:>8.0f} {sl.iloc[i]:>8.0f} {adx.iloc[i]:>5.1f} {adx_chg:>+6.1f} {pdi.iloc[i]:>5.1f} {mdi.iloc[i]:>5.1f} {'✓' if pdi.iloc[i] > mdi.iloc[i] else '✗':>8} {'✓' if adx.iloc[i] > 25 else '✗':>7} {label:>6}")
