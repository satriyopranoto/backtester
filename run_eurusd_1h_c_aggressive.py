"""
EURUSD 1H — C_Aggressive Intraday Parameters
Single test: ADX=7, SL=5, SMA=10, SLx=2.2, PDIbars=3
"""
import sys
sys.path.insert(0, r'C:\Users\satri\code\backtester')

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from strategies.bb_adx_strategy import donchian_sl, calc_adx
import warnings
warnings.filterwarnings("ignore")

# Monkey-patch for pandas 3.x compatibility (is_numeric removed)
if not hasattr(pd.Index, 'is_numeric'):
    pd.Index.is_numeric = property(lambda self: self.dtype.kind in 'iufcb')

# ── Fast Parameters (C_Aggressive) ──
ADX_PERIOD = 7
SL_PERIOD = 5
SL_MULTIPLE = 2.2
BB_PERIOD = 10   # SMA basis
RISK_PCT = 1.0
TP_RATIO = 0.4
MIN_ADX = 20
PDI_BARS = 3

class C_Aggressive_EURUSD(Strategy):
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
        
        self.basis = self.I(_basis, self.data.Close, self.bb_period,
                           name=f"Basis({self.bb_period})", overlay=True)
        self.sl = self.I(donchian_sl, self.data.High, self.data.Low,
                        self.sl_multiple, self.sl_period,
                        name="SL", overlay=True)
        self.adx_arr, self.pdi_arr, self.mdi_arr = calc_adx(
            np.asarray(self.data.High), np.asarray(self.data.Low),
            np.asarray(self.data.Close), self.adx_period)

    def next(self):
        idx = len(self.data) - 1
        close = float(self.data.Close[-1])
        low = float(self.data.Low[-1])
        basis = float(self.basis[-1])
        sl = float(self.sl[-1])
        adx = float(self.adx_arr[idx])
        pdi = float(self.pdi_arr[idx])
        mdi = float(self.mdi_arr[idx])
        
        nan = np.isnan(adx) or np.isnan(pdi) or np.isnan(mdi) or np.isnan(basis) or np.isnan(sl)
        
        # ── EXIT ──
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
        
        # ── ENTRY ──
        if nan: return
        if not (low > sl and close > basis): return
        if not (adx > MIN_ADX): return
        if not (pdi > mdi): return
        if idx >= PDI_BARS:
            pdi_ago = float(self.pdi_arr[idx - PDI_BARS])
            if not (pdi > pdi_ago): return
        
        stop_dist = abs(close - sl)
        if stop_dist <= 0: return
        risk_amount = self.equity * (self.risk_pct / 100.0)
        size = int(risk_amount / stop_dist)
        max_by_cash = int((self.equity * 0.95) / close)
        if max_by_cash < 1: return
        size = max(1, min(size, max_by_cash))
        
        if size > 0:
            self.buy(size=size)
            self._entry_sl = sl
            self._entry_price = close
            self._tp_threshold_pct = (stop_dist / close) * 100.0 * TP_RATIO


# ── Load data ──
print("=" * 65)
print("  EURUSD 1H — C_Aggressive (ADX=7 SL=5 SMA=10)")
print("=" * 65)

data = pd.read_csv(r'C:\Users\satri\code\backtester\EURUSD_1h_yf.txt')
data['Datetime'] = pd.to_datetime(data['Datetime'], utc=True)
data = data.rename(columns={'Datetime': 'Date'}).set_index('Date').sort_index()
data.index = data.index.tz_localize(None)
data.columns = [c.strip() for c in data.columns]
print(f"  Data: {len(data)} bars, {data.index[0]} → {data.index[-1]}")

bt = Backtest(data, C_Aggressive_EURUSD, cash=100_000, commission=0.001)
stats = bt.run()

# ── Print results ──
trades = stats['_trades']
n_trades = len(trades) if isinstance(trades, pd.DataFrame) else 0

print()
print("=" * 65)
print("  EURUSD 1H — C_AGGRESSIVE RESULTS")
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

# Save report
from backtesting.lib import plot_heatmaps
try:
    bt.plot(filename=r'C:\Users\satri\code\backtester\reports\EURUSD_1h_c_aggressive.html', open_browser=False)
    print(f"\n  📁 Report: reports/EURUSD_1h_c_aggressive.html")
except:
    pass

# Save trades
if isinstance(trades, pd.DataFrame) and len(trades) > 0:
    trades.to_csv(r'C:\Users\satri\code\backtester\reports\EURUSD_1h_c_aggressive_trades.csv')
    print(f"  📁 Trades saved")
    sl_count = len(trades[trades['ExitReason'] == 'SL']) if 'ExitReason' in trades.columns else '?'
    tp_count = len(trades[trades['ExitReason'] == 'TP']) if 'ExitReason' in trades.columns else '?'
    print(f"  SL/TP: {sl_count}/{tp_count}")
