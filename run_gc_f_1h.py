"""Basis ADX 1H — Gold Futures (GC=F), 1 year data, same-TF SL (0.4R, no SMA200)."""
import pandas as pd
import numpy as np
from pathlib import Path
from backtesting import Backtest, Strategy
from strategies.bb_adx_strategy import calc_adx, donchian_sl
import warnings; warnings.filterwarnings("ignore")

BASE = Path(r"C:\Users\satri\code\backtester")

# ── Load data ──
raw = pd.read_csv(BASE / "GC_F_1h_yf.txt")
raw["Datetime"] = pd.to_datetime(raw["Datetime"], utc=True)
raw = raw.rename(columns={"Datetime": "Date"}).set_index("Date").sort_index()
raw.index = raw.index.tz_localize(None)
keep = ["Open", "High", "Low", "Close", "Volume"]
raw = raw[[c for c in keep if c in raw.columns]]

print(f"  Bars: {len(raw)} ({raw.index[0]} — {raw.index[-1]})")

# ── Compute Donchian SL ──
SL_MULTIPLE = 2.8; SL_PERIOD = 10
sl_arr = donchian_sl(raw["High"].values, raw["Low"].values, SL_MULTIPLE, SL_PERIOD)
raw["sl"] = sl_arr
raw = raw.dropna(subset=["sl"])

print(f"  After drop NaN SL: {len(raw)} bars ({raw.index[0]} — {raw.index[-1]})")

# ── Strategy ──
class BasisAdxGold(Strategy):
    bb_period = 20; adx_period = 14; risk_pct = 1.0
    _entry_sl: float | None = None; _entry_price: float | None = None; _tp_threshold_pct: float | None = None

    def init(self) -> None:
        def _basis(arr, p): return pd.Series(arr).rolling(p).mean().values
        self.basis = self.I(_basis, self.data.Close, self.bb_period,
                            name=f"Basis({self.bb_period})", overlay=True)
        self.sl_line = self.I(lambda: self.data.sl, name="SL (Donchian)",
                              overlay=True, color="orange")
        self.adx_arr, self.pdi_arr, self.mdi_arr = calc_adx(
            np.asarray(self.data.High), np.asarray(self.data.Low),
            np.asarray(self.data.Close), self.adx_period)
        self.I(lambda: pd.DataFrame({'ADX': self.adx_arr, '+DI': self.pdi_arr, '-DI': self.mdi_arr}),
               name="ADX/DI", overlay=False)

    def next(self) -> None:
        idx = len(self.data) - 1
        c = float(self.data.Close[-1]); l = float(self.data.Low[-1])
        b = float(self.basis[-1]); s = float(self.data.sl[-1])
        a = float(self.adx_arr[idx]); p = float(self.pdi_arr[idx])
        m = float(self.mdi_arr[idx]); p5 = float(self.pdi_arr[idx-5]) if idx >= 5 else 0.0
        nan = np.isnan(a) or np.isnan(p) or np.isnan(m) or np.isnan(s)

        # EXIT
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

        # ENTRY
        if (l > s and c > b and not nan and a > 20.0 and p > m and p > p5):
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

# ── Run ──
print("\n" + "=" * 55)
print("  🚀 GC=F 1H — Basis ADX (0.4R, no SMA200)")
print("=" * 55)

bt = Backtest(raw, BasisAdxGold, cash=100_000, commission=0.0001, finalize_trades=True)
stats = bt.run()

print(f"\n  Periode      : {stats['Start'].date()} — {stats['End'].date()}")
print(f"  Return       : {stats['Return [%]']:.2f}%")
print(f"  Buy & Hold   : {stats['Buy & Hold Return [%]']:.2f}%")
print(f"  Equity Final : ${stats['Equity Final [$]']:,.2f}")
print(f"  Equity Peak  : ${stats['Equity Peak [$]']:,.2f}")
print(f"  Sharpe       : {stats['Sharpe Ratio']:.2f}")
print(f"  Max DD       : {stats['Max. Drawdown [%]']:.2f}%")
print(f"  # Trades     : {stats['# Trades']}")
print(f"  Win Rate     : {stats.get('Win Rate [%]', 0):.1f}%")
print(f"  Best Trade   : {stats.get('Best Trade [%]', 0):.2f}%")
print(f"  Worst Trade  : {stats.get('Worst Trade [%]', 0):.2f}%")
print(f"  Avg Trade    : {stats.get('Avg Trade [%]', 0):.2f}%")
print(f"  Avg Duration : {stats.get('Avg. Duration [%]', 0):.2f}%")

# ── Save HTML ──
rpt = BASE / "reports" / "GC_F_basis_adx_1h.html"
bt.plot(filename=str(rpt), open_browser=False)
print(f"\n  📄 Report: {rpt}")
print("=" * 55)
print("✅ DONE")
