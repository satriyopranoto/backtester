"""Backtest EURUSD — compare 1H vs 4H vs 1D — Basis ADX (0.4R, no SMA200)."""
import pandas as pd
import numpy as np
from pathlib import Path
from backtesting import Backtest, Strategy
from strategies.bb_adx_strategy import calc_adx, donchian_sl
import warnings; warnings.filterwarnings("ignore")
import yfinance as yf

BASE = Path(r"C:\Users\satri\code\backtester")
SL_MULTIPLE = 2.8; SL_PERIOD = 10

def load_data(ticker, interval, period):
    df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
    if df.columns.nlevels > 1:
        df.columns = [c[0] for c in df.columns]
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    keep = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]]
    return df.dropna()

class BasisAdx(Strategy):
    bb_period = 20; adx_period = 14; risk_pct = 1.0
    _entry_sl = None; _entry_price = None; _tp_threshold_pct = None

    def init(self):
        def _b(arr, p): return pd.Series(arr).rolling(p).mean().values
        self.basis = self.I(_b, self.data.Close, self.bb_period, name="Basis", overlay=True)
        self.sl_line = self.I(lambda: self.data.sl, name="SL", overlay=True, color="orange")
        self.adx_arr, self.pdi_arr, self.mdi_arr = calc_adx(
            np.asarray(self.data.High), np.asarray(self.data.Low),
            np.asarray(self.data.Close), self.adx_period)

    def next(self):
        idx = len(self.data) - 1
        c = float(self.data.Close[-1]); l = float(self.data.Low[-1])
        b = float(self.basis[-1]); s = float(self.data.sl[-1])
        a = float(self.adx_arr[idx]); p = float(self.pdi_arr[idx])
        m = float(self.mdi_arr[idx]); p5 = float(self.pdi_arr[idx-5]) if idx >= 5 else 0.0
        nan = np.isnan(a) or np.isnan(p) or np.isnan(m) or np.isnan(s)

        if self._entry_sl is not None and self._entry_price is not None:
            if c < self._entry_sl:
                self.position.close()
                self._entry_sl = None; self._entry_price = None; self._tp_threshold_pct = None
                return
            tp = self._tp_threshold_pct if self._tp_threshold_pct is not None else 999.0
            if self._entry_price > 0:
                fp = ((c - self._entry_price) / self._entry_price) * 100.0
                if fp > tp and c < s:
                    self.position.close()
                    self._entry_sl = None; self._entry_price = None; self._tp_threshold_pct = None
                    return
            return

        if (l > s and c > b and not nan and a > 20 and p > m and p > p5):
            sd = abs(c - s)
            if sd > 0:
                ra = self.equity * (self.risk_pct / 100.0); sz = int(ra / sd)
                mbc = int((self.equity * 0.95) / c)
                if mbc >= 1: sz = max(1, sz)
                sz = min(sz, mbc)
            else: sz = 0
            if sz > 0:
                self.buy(size=sz)
                self._entry_sl = s; self._entry_price = c
                self._tp_threshold_pct = (sd / c) * 100.0 * 0.4

def run_bt(name, df, sl_multiple=SL_MULTIPLE, sl_period=SL_PERIOD):
    sl_arr = donchian_sl(df["High"].values, df["Low"].values, sl_multiple, sl_period)
    d = df.copy()
    d["sl"] = sl_arr
    d = d.dropna(subset=["sl"])
    
    bt = Backtest(d, BasisAdx, cash=100_000, commission=0.0001)
    s = bt.run()
    rpt = BASE / "reports" / f"EURUSD_{name}.html"
    bt.plot(filename=str(rpt), open_browser=False)
    
    return s, len(d), rpt

# ── DOWNLOAD DATA ──
print("📥 Downloading data...")
d1h = load_data("EURUSD=X", "1h", "1y")
d4h = load_data("EURUSD=X", "4h", "2y")
d1d = load_data("EURUSD=X", "1d", "5y")

print(f"  1H: {len(d1h)} bars")
print(f"  4H: {len(d4h)} bars")
print(f"  1D: {len(d1d)} bars")

# ── RUN ──
results = []
for name, df, period_label in [
    ("1h", d1h, "1 year"),
    ("4h", d4h, "2 years"),
    ("1d", d1d, "5 years"),
]:
    print(f"\n🚀 Running {name}...")
    s, bars, rpt = run_bt(name, df)
    results.append((name, period_label, bars, s, rpt))
    
    print(f"  Bars      : {bars}")
    print(f"  Period    : {s['Start'].date()} — {s['End'].date()}")
    print(f"  Return    : {s['Return [%]']:.2f}%")
    print(f"  Sharpe    : {s['Sharpe Ratio']:.2f}")
    print(f"  Max DD    : {s['Max. Drawdown [%]']:.2f}%")
    print(f"  Trades    : {s['# Trades']}")
    print(f"  Win Rate  : {s.get('Win Rate [%]', 0):.1f}%")
    print(f"  Best      : {s.get('Best Trade [%]', 0):.2f}%")
    print(f"  Worst     : {s.get('Worst Trade [%]', 0):.2f}%")

# ── SUMMARY ──
print("\n\n" + "=" * 60)
print("  📊 SUMMARY: EURUSD — BASIS ADX (0.4R, No SMA200)")
print("=" * 60)
print(f"  {'TF':<5} {'Period':<12} {'Bars':<7} {'Return':<9} {'Sharpe':<8} {'DD':<8} {'Trades':<8} {'Win%':<6} {'Best':<8} {'Worst':<8}")
print(f"  {'─'*4} {'─'*11} {'─'*6} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*5} {'─'*7} {'─'*7}")

for name, period_label, bars, s, rpt in results:
    print(f"  {name:<5} {period_label:<12} {bars:<7} {s['Return [%]']:>7.2f}% {s['Sharpe Ratio']:>6.2f} {s['Max. Drawdown [%]']:>6.2f}% {s['# Trades']:>6d} {s.get('Win Rate [%]', 0):>4.1f}% {s.get('Best Trade [%]', 0):>6.2f}% {s.get('Worst Trade [%]', 0):>6.2f}%")

print(f"\n  {'='*60}")
winner = max(results, key=lambda r: r[3]['Return [%]'])
print(f"  🏆 Winner: {winner[0]} ({winner[3]['Return [%]']:.2f}%)")
print(f"  {'='*60}")
print("✅ DONE")
