"""
AUDJPY 1H — Normal Daily Parameters (ADX=14, SL=10, SMA=20, SLx=2.8)
"""
import sys
sys.path.insert(0, r'C:\Users\satri\code\backtester')

import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
from strategies.bb_adx_strategy import donchian_sl, calc_adx
import warnings
warnings.filterwarnings("ignore")

# Monkey-patch for pandas 3.x
if not hasattr(pd.Index, 'is_numeric'):
    pd.Index.is_numeric = property(lambda self: self.dtype.kind in 'iufcb')

# ── Normal Daily Parameters ──
ADX_PERIOD = 14
SL_PERIOD = 10
SL_MULTIPLE = 2.8
BB_PERIOD = 20
RISK_PCT = 1.0
TP_RATIO = 0.4
MIN_ADX = 20
PDI_BARS = 5

class NormalStrategy(Strategy):
    bb_period = BB_PERIOD
    adx_period = ADX_PERIOD
    sl_multiple = SL_MULTIPLE
    sl_period = SL_PERIOD
    risk_pct = RISK_PCT
    
    _entry_sl: float | None = None
    _entry_price: float | None = None
    _tp_threshold_pct: float | None = None

    def init(self):
        def _basis(arr, period):
            return pd.Series(arr).rolling(period).mean().values
        self.basis = self.I(_basis, self.data.Close, self.bb_period, name=f"Basis({self.bb_period})", overlay=True)
        self.sl = self.I(donchian_sl, self.data.High, self.data.Low, self.sl_multiple, self.sl_period, name="SL", overlay=True)
        self.adx_arr, self.pdi_arr, self.mdi_arr = calc_adx(
            np.asarray(self.data.High), np.asarray(self.data.Low), np.asarray(self.data.Close), self.adx_period)

    def next(self):
        idx = len(self.data) - 1
        close = float(self.data.Close[-1]); low = float(self.data.Low[-1])
        basis = float(self.basis[-1]); sl = float(self.sl[-1])
        adx = float(self.adx_arr[idx]); pdi = float(self.pdi_arr[idx]); mdi = float(self.mdi_arr[idx])
        nan = np.isnan(adx) or np.isnan(pdi) or np.isnan(mdi) or np.isnan(basis) or np.isnan(sl)
        
        if self._entry_sl is not None:
            if close < self._entry_sl:
                self.position.close()
                self._entry_sl = None; self._entry_price = None; self._tp_threshold_pct = None
                return
            if self._tp_threshold_pct is not None and self._entry_price > 0:
                fl = ((close - self._entry_price) / self._entry_price) * 100.0
                if fl > self._tp_threshold_pct:
                    self.position.close()
                    self._entry_sl = None; self._entry_price = None; self._tp_threshold_pct = None
                    return
            return
        
        if nan: return
        if not (low > sl and close > basis): return
        if not (adx > MIN_ADX): return
        if not (pdi > mdi): return
        if idx >= PDI_BARS:
            pdi_ago = float(self.pdi_arr[idx - PDI_BARS])
            if not (pdi > pdi_ago): return
        
        stop_dist = abs(close - sl)
        if stop_dist <= 0: return
        size = int(self.equity * (self.risk_pct / 100.0) / stop_dist)
        max_sz = int((self.equity * 0.95) / close)
        if max_sz < 1: return
        size = max(1, min(size, max_sz))
        
        if size > 0:
            self.buy(size=size)
            self._entry_sl = sl; self._entry_price = close
            self._tp_threshold_pct = (stop_dist / close) * 100.0 * TP_RATIO


# ── Download AUDJPY 1h from Yahoo ──
print("=" * 65)
print("  Downloading AUDJPY 1h from Yahoo Finance...")
print("=" * 65)

df = yf.download("AUDJPY=X", period="1y", interval="1h", progress=False, auto_adjust=True)
if df.empty:
    print("  No data for AUDJPY=X, trying alternate symbol...")
    df = yf.download("AUDJPY", period="1y", interval="1h", progress=False, auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.columns = [c.strip() for c in df.columns]
print(f"  Downloaded: {len(df)} bars, {df.index[0]} → {df.index[-1]}")

# Convert to backtesting format
data = df.rename(columns={'Close':'Close','High':'High','Low':'Low','Open':'Open','Volume':'Volume'})
data.index.name = 'Date'
data.index = data.index.tz_localize(None)
data = data.dropna()
print(f"  Cleaned: {len(data)} bars")

print()
print("=" * 65)
print(f"  AUDJPY 1H — Normal Daily Parameters")
print(f"  ADX={ADX_PERIOD} SL={SL_PERIOD} SMA={BB_PERIOD} SLx={SL_MULTIPLE} PDIb={PDI_BARS}")
print("=" * 65)

bt = Backtest(data, NormalStrategy, cash=100_000, commission=0.001)
stats = bt.run()

trades = stats['_trades']
n_trades = len(trades) if isinstance(trades, pd.DataFrame) else 0

print()
print("=" * 65)
print("  AUDJPY 1H — NORMAL PARAMS RESULTS")
print("=" * 65)
print(f"  Period        : {data.index[0].date()} → {data.index[-1].date()}")
print(f"  Total Bars    : {len(data)}")
print(f"  Parameters    : ADX={ADX_PERIOD} SL={SL_PERIOD} SMA={BB_PERIOD} SLx={SL_MULTIPLE}")
print(f"  Initial Capital: $100,000")
print(f"  Final Equity  : ${stats['Equity Final [$]']:,.0f}")
print(f"  Total Return  : {stats['Return [%]']:+.2f}%")
print(f"  CAGR          : {stats.get('CAGR [%]', 0):+.2f}%")
print(f"  Max DD        : {stats['Max. Drawdown [%]']:.2f}%")
print(f"  Sharpe        : {stats['Sharpe Ratio']:.2f}")
print(f"  Profit Factor : {stats['Profit Factor']:.2f}")
print(f"  Total Trades  : {n_trades}")
print(f"  Win Rate      : {stats['Win Rate [%]']:.1f}%")
print(f"  Avg Trade     : {stats['Avg. Trade [%]']:.2f}%")
print(f"  Best Trade    : {stats['Best Trade [%]']:.2f}%")
print(f"  Worst Trade   : {stats['Worst Trade [%]']:.2f}%")
print("=" * 65)

try:
    bt.plot(filename=r'C:\Users\satri\code\backtester\reports\AUDJPY_1h_normal.html', open_browser=False)
    print(f"\n  📁 Report: reports/AUDJPY_1h_normal.html")
except: pass

if isinstance(trades, pd.DataFrame) and len(trades) > 0:
    trades.to_csv(r'C:\Users\satri\code\backtester\reports\AUDJPY_1h_normal_trades.csv')
    print(f"  📁 Trades saved")
