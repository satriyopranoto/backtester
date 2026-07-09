"""Compare: Original vs Reversal — XAUUSD 1h."""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy

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
    adx = pd.Series(dx).rolling(14).mean().values
    return adx, pdi, mdi

def donchian_sl(high, low, mul=2.8, period=10):
    ero = int(mul * period)
    s_high, s_low = pd.Series(high), pd.Series(low)
    r_prev = s_high.rolling(ero).max().shift(1).values
    s_prev = s_low.rolling(ero).min().shift(1).values
    r_curr = s_high.rolling(ero).max().values
    s_curr = s_low.rolling(ero).min().values
    ab = np.where(high > r_prev, 1, np.where(low < s_prev, -1, 0))
    ac = pd.Series(ab).replace(0, np.nan).ffill().fillna(0).values
    return np.where(ac == 1, s_curr, r_curr)

# ── Download XAUUSD 1h ──
print("Downloading XAUUSD 1h...")
h1 = yf.download('GC=F', period='2y', interval='1h', progress=False, auto_adjust=True)
if isinstance(h1.columns, pd.MultiIndex):
    h1.columns = h1.columns.get_level_values(0)
h1.columns = [c.lower() for c in h1.columns]
h1 = h1.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
h1.index = pd.to_datetime(h1.index)
if h1.index.tz is not None:
    h1.index = h1.index.tz_localize(None)
print(f"  {len(h1)} bars | {h1.index[0]} → {h1.index[-1]}")

# ── Download & map daily HTF ──
print("Downloading XAUUSD Daily...")
daily = yf.download('GC=F', period='5y', interval='1d', progress=False, auto_adjust=True)
if isinstance(daily.columns, pd.MultiIndex):
    daily.columns = daily.columns.get_level_values(0)
daily.columns = [c.lower() for c in daily.columns]
daily.index = pd.to_datetime(daily.index)
if daily.index.tz is not None:
    daily.index = daily.index.tz_localize(None)

dc, dh, dl = daily['close'].values, daily['high'].values, daily['low'].values
_, dp, dm = calc_adx(dc, dh, dl)

h1['date'] = h1.index.normalize()
daily_map = pd.DataFrame({'date': daily.index, 'htf_pdi': dp, 'htf_mdi': dm}).set_index('date')
h1 = pd.merge_asof(h1.sort_values('date'), daily_map, left_on='date', right_index=True, direction='backward')
h1 = h1.dropna(subset=['htf_pdi','htf_mdi']).drop(columns=['date'])
print(f"  {len(h1)} bars with HTF data")

# ── Pre-compute indicators ──
c = h1['Close'].values.astype(float)
hi = h1['High'].values.astype(float)
lo = h1['Low'].values.astype(float)
adx, pdi, mdi = calc_adx(c, hi, lo)
sl_arr = donchian_sl(hi, lo)
sma20 = pd.Series(c).rolling(20).mean().values

# ── Strategy A: Original BUY only ──
class OriginalXau(Strategy):
    def init(self):
        self.sl_line = self.I(lambda: sl_arr, name='SL', overlay=True)
        self.sma = self.I(lambda: sma20, name='SMA20', overlay=True)
        self.adx_i = self.I(lambda: adx, name='ADX', overlay=False)
        self.pdi_i = self.I(lambda: pdi, name='+DI', overlay=False)
        self.mdi_i = self.I(lambda: mdi, name='-DI', overlay=False)

    def next(self):
        i = len(self.data) - 1
        if i < 28: return
        close, low = float(self.data.Close[-1]), float(self.data.Low[-1])
        sl_v = float(self.sl_line[-1])
        if np.isnan(sl_v) or sl_v <= 0: return

        # Exit
        if self.position:
            if self._entry_sl is not None and close < self._entry_sl:
                self.position.close()
                self._entry_sl = None; self._entry_price = None; self._tp_pct = None
                return
            if self._entry_price and self._tp_pct:
                fl = ((close - self._entry_price) / self._entry_price) * 100.0
                if fl > self._tp_pct and close < sl_v:
                    self.position.close()
                    self._entry_sl = None; self._entry_price = None; self._tp_pct = None
                    return
            return

        # Entry
        a, p, m = float(adx[i]), float(pdi[i]), float(mdi[i])
        if np.isnan(a) or np.isnan(p) or np.isnan(m): return
        a5 = float(adx[i-5]) if i >= 5 else 0
        p5 = float(pdi[i-5]) if i >= 5 else 0
        dp_i = float(h1['htf_pdi'].iloc[i])
        dm_i = float(h1['htf_mdi'].iloc[i])

        if (low > sl_v and close > sma20[i] and
            a > 20 and a > a5 and p > m and p > p5 and
            not np.isnan(dp_i) and not np.isnan(dm_i) and dp_i > dm_i):
            sd = abs(close - sl_v)
            if sd > 0:
                sz = max(1, int((self.equity * 0.01) / sd))
                mxc = int((self.equity * 0.95) / close)
                sz = min(sz, mxc)
                if sz > 0:
                    self.buy(size=sz)
                    self._entry_sl = sl_v
                    self._entry_price = close
                    self._tp_pct = (sd / close) * 100.0 * 0.4

OriginalXau._entry_sl = None; OriginalXau._entry_price = None; OriginalXau._tp_pct = None


# ── Run both ──
results = {}
for name, strat, label in [
    ("Original", OriginalXau, "Original Multi TF (BUY only)"),
    ("Reversal", BasisAdxMultiTfReversal, "Reversal (LONG+SHORT)"),
]:
    print(f"\n  ⏳ Running {label}...")
    bt = Backtest(h1, strat, cash=100_000, commission=0.001)
    s = bt.run()
    results[name] = s

    trades = s['_trades']
    n_long = len(trades[trades['Size'] > 0]) if trades is not None and len(trades) > 0 else 0
    n_short = len(trades[trades['Size'] < 0]) if trades is not None and len(trades) > 0 else 0

    print(f"  {'='*40}")
    print(f"  {label}")
    print(f"  {'='*40}")
    print(f"  Return    : {s['Return [%]']:+.2f}%")
    print(f"  Buy&Hold  : {s['Buy & Hold Return [%]']:+.2f}%")
    print(f"  Trades    : {s['# Trades']} ({n_long}L / {n_short}S)")
    print(f"  Win Rate  : {s['Win Rate [%]']:.1f}%")
    print(f"  Max DD    : -{s['Max. Drawdown [%]']:.2f}%")
    print(f"  Sharpe    : {s['Sharpe Ratio']:.2f}")
    print(f"  Best/Worst: +{s['Best Trade [%]']:.2f}% / {s['Worst Trade [%]']:.2f}%")

print(f"\n\n{'='*50}")
print(f"  COMPARISON — XAUUSD H1")
print(f"{'='*50}")
print(f"  {'Metric':<20} {'Original':>12} {'Reversal':>12}")
print(f"  {'-'*20} {'-'*12} {'-'*12}")
r_o, r_r = results["Original"], results["Reversal"]
print(f"  {'Return':<20} {r_o['Return [%]']:>+11.2f}% {r_r['Return [%]']:>+11.2f}%")
print(f"  {'Buy & Hold':<20} {r_o['Buy & Hold Return [%]']:>+11.2f}% {r_r['Buy & Hold Return [%]']:>+11.2f}%")
print(f"  {'Trades':<20} {r_o['# Trades']:>11d} {r_r['# Trades']:>11d}")
print(f"  {'Win Rate':<20} {r_o['Win Rate [%]']:>10.1f}% {r_r['Win Rate [%]']:>10.1f}%")
print(f"  {'Max DD':<20} -{r_o['Max. Drawdown [%]']:>9.2f}% -{r_r['Max. Drawdown [%]']:>9.2f}%")
print(f"  {'Sharpe':<20} {r_o['Sharpe Ratio']:>11.2f} {r_r['Sharpe Ratio']:>11.2f}")
print(f"  {'Best Trade':<20} +{r_o['Best Trade [%]']:>10.2f}% +{r_r['Best Trade [%]']:>10.2f}%")
print(f"  {'Worst Trade':<20} {r_o['Worst Trade [%]']:>11.2f}% {r_r['Worst Trade [%]']:>11.2f}%")
