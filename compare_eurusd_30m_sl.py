"""Compare EURUSD 30min: Same-TF SL vs Daily SL — Basis ADX (0.4R, no SMA200)."""
import pandas as pd
import numpy as np
from pathlib import Path
from backtesting import Backtest, Strategy
from strategies.bb_adx_strategy import calc_adx, donchian_sl
import warnings; warnings.filterwarnings("ignore")

BASE = Path(r"C:\Users\satri\code\backtester")
SL_MULTIPLE = 2.8
SL_PERIOD   = 10

# ── 1. Load 30min data ──
def load_30m():
    df = pd.read_csv(BASE / "EURUSD_30m_yf.txt")
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    df = df.rename(columns={"Datetime": "Date"}).set_index("Date").sort_index()
    df.index = df.index.tz_localize(None)
    keep = ["Open", "High", "Low", "Close", "Volume"]
    return df[[c for c in keep if c in df.columns]]

# ── 2. Build daily SL map ──
def build_daily_sl_map():
    daily = pd.read_csv(BASE / "EURUSD_1d_yf_30m.txt")
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.set_index("Date").sort_index()

    sl_arr = donchian_sl(daily["High"].values, daily["Low"].values, SL_MULTIPLE, SL_PERIOD)
    daily["daily_sl"] = sl_arr
    daily["daily_sl"] = daily["daily_sl"].shift(1)

    sl_map = {}
    for k, v in daily["daily_sl"].to_dict().items():
        if pd.notna(v) and not np.isnan(v):
            sl_map[k.date()] = v
    return sl_map

# ── Strategy A: Same-TF SL ──
class BasisAdxSameSL(Strategy):
    """Basis ADX with same-TF Donchian SL."""
    bb_period = 20; adx_period = 14; risk_pct = 1.0
    _entry_sl: float | None = None
    _entry_price: float | None = None
    _tp_threshold_pct: float | None = None

    def init(self) -> None:
        def _basis(arr, p): return pd.Series(arr).rolling(p).mean().values
        self.basis = self.I(_basis, self.data.Close, self.bb_period,
                            name=f"Basis({self.bb_period})", overlay=True)
        self.sl_line = self.I(lambda: self.data.sl,
                              name="SL (30m Donchian)", overlay=True, color="orange")
        self.adx_arr, self.pdi_arr, self.mdi_arr = calc_adx(
            np.asarray(self.data.High), np.asarray(self.data.Low),
            np.asarray(self.data.Close), self.adx_period)
        self.I(lambda: pd.DataFrame({'ADX': self.adx_arr, '+DI': self.pdi_arr, '-DI': self.mdi_arr}),
               name="ADX/DI", overlay=False)

    def next(self) -> None:
        self._next_impl("same")

    def _next_impl(self, src: str) -> None:
        idx = len(self.data) - 1
        close = float(self.data.Close[-1]); low = float(self.data.Low[-1])
        basis = float(self.basis[-1])
        sl_val = float(self.data.sl[-1]) if src == "same" else float(self.data.daily_sl[-1])
        adx = float(self.adx_arr[idx]); pdi = float(self.pdi_arr[idx])
        mdi = float(self.mdi_arr[idx]); pdi_5ago = float(self.pdi_arr[idx-5]) if idx >= 5 else 0.0
        is_nan = np.isnan(adx) or np.isnan(pdi) or np.isnan(mdi) or np.isnan(sl_val)

        if self._entry_sl is not None and self._entry_price is not None:
            if close < self._entry_sl:
                self.position.close(); self._entry_sl = None; self._entry_price = None; self._tp_threshold_pct = None
                return
            tp_th = self._tp_threshold_pct if self._tp_threshold_pct is not None else 999.0
            if self._entry_price > 0:
                fp = ((close - self._entry_price) / self._entry_price) * 100.0
                if fp > tp_th and close < sl_val:
                    self.position.close(); self._entry_sl = None; self._entry_price = None; self._tp_threshold_pct = None
                    return
            return

        if (low > sl_val and close > basis and not is_nan and
            adx > 20.0 and pdi > mdi and pdi > pdi_5ago):
            sd = abs(close - sl_val)
            if sd > 0:
                ra = self.equity * (self.risk_pct / 100.0)
                sz = int(ra / sd); mbc = int((self.equity * 0.95) / close)
                if mbc >= 1: sz = max(1, sz)
                sz = min(sz, mbc)
            else: sz = 0
            if sz > 0:
                self.buy(size=sz); self._entry_sl = sl_val; self._entry_price = close
                self._tp_threshold_pct = (sd / close) * 100.0 * 0.4


# ── Strategy B: Daily SL ──
class BasisAdxDailySL(Strategy):
    """Basis ADX with daily Donchian SL."""
    bb_period = 20; adx_period = 14; risk_pct = 1.0
    _entry_sl: float | None = None
    _entry_price: float | None = None
    _tp_threshold_pct: float | None = None

    def init(self) -> None:
        def _basis(arr, p): return pd.Series(arr).rolling(p).mean().values
        self.basis = self.I(_basis, self.data.Close, self.bb_period,
                            name=f"Basis({self.bb_period})", overlay=True)
        self.sl_line = self.I(lambda: self.data.daily_sl,
                              name="SL (Daily Donchian)", overlay=True, color="red")
        self.adx_arr, self.pdi_arr, self.mdi_arr = calc_adx(
            np.asarray(self.data.High), np.asarray(self.data.Low),
            np.asarray(self.data.Close), self.adx_period)
        self.I(lambda: pd.DataFrame({'ADX': self.adx_arr, '+DI': self.pdi_arr, '-DI': self.mdi_arr}),
               name="ADX/DI", overlay=False)

    def next(self) -> None:
        self._next_impl("daily")

    def _next_impl(self, src: str) -> None:
        idx = len(self.data) - 1
        close = float(self.data.Close[-1]); low = float(self.data.Low[-1])
        basis = float(self.basis[-1])
        sl_val = float(self.data.daily_sl[-1]) if src == "daily" else float(self.data.sl[-1])
        adx = float(self.adx_arr[idx]); pdi = float(self.pdi_arr[idx])
        mdi = float(self.mdi_arr[idx]); pdi_5ago = float(self.pdi_arr[idx-5]) if idx >= 5 else 0.0
        is_nan = np.isnan(adx) or np.isnan(pdi) or np.isnan(mdi) or np.isnan(sl_val)

        if self._entry_sl is not None and self._entry_price is not None:
            if close < self._entry_sl:
                self.position.close(); self._entry_sl = None; self._entry_price = None; self._tp_threshold_pct = None
                return
            tp_th = self._tp_threshold_pct if self._tp_threshold_pct is not None else 999.0
            if self._entry_price > 0:
                fp = ((close - self._entry_price) / self._entry_price) * 100.0
                if fp > tp_th and close < sl_val:
                    self.position.close(); self._entry_sl = None; self._entry_price = None; self._tp_threshold_pct = None
                    return
            return

        if (low > sl_val and close > basis and not is_nan and
            adx > 20.0 and pdi > mdi and pdi > pdi_5ago):
            sd = abs(close - sl_val)
            if sd > 0:
                ra = self.equity * (self.risk_pct / 100.0)
                sz = int(ra / sd); mbc = int((self.equity * 0.95) / close)
                if mbc >= 1: sz = max(1, sz)
                sz = min(sz, mbc)
            else: sz = 0
            if sz > 0:
                self.buy(size=sz); self._entry_sl = sl_val; self._entry_price = close
                self._tp_threshold_pct = (sd / close) * 100.0 * 0.4


# ── Main ──
# Load 30m base data
h30 = load_30m()
print("=" * 55)
print(f"  EURUSD 30min — {len(h30)} bars")
print(f"  {h30.index[0]} — {h30.index[-1]}")
print("=" * 55)

# ── TEST A: Same-TF SL ──
print("\n📊 [A] SL SAME TIMEFRAME (30m Donchian)")
print("-" * 40)

# Compute same-TF Donchian SL
ero = int(SL_MULTIPLE * SL_PERIOD)
sl_arr = donchian_sl(h30["High"].values, h30["Low"].values, SL_MULTIPLE, SL_PERIOD)
h30_same = h30.copy()
h30_same["sl"] = sl_arr
h30_same = h30_same.dropna(subset=["sl"])

print(f"  Bars after NaN drop: {len(h30_same)}")
print(f"  Range: {h30_same.index[0]} — {h30_same.index[-1]}")

bt_a = Backtest(h30_same, BasisAdxSameSL, cash=100_000, commission=0.0001, finalize_trades=True)
stats_a = bt_a.run()
rpt_a = BASE / "reports" / "EURUSD_30m_same_sl.html"
bt_a.plot(filename=str(rpt_a), open_browser=False)
print(f"  Return : {stats_a['Return [%]']:.2f}%")
print(f"  Sharpe : {stats_a['Sharpe Ratio']:.2f}")
print(f"  Max DD : {stats_a['Max. Drawdown [%]']:.2f}%")
print(f"  Trades : {stats_a['# Trades']}")
print(f"  Win%   : {stats_a.get('Win Rate [%]', 0):.1f}%")
print(f"  Best   : {stats_a.get('Best Trade [%]', 0):.2f}%")
print(f"  Worst  : {stats_a.get('Worst Trade [%]', 0):.2f}%")
bt_rpt_a = BASE / "reports" / "EURUSD_30m_same_sl.html"

# ── TEST B: Daily SL ──
print("\n📊 [B] SL DAILY (Daily Donchian)")
print("-" * 40)

sl_map = build_daily_sl_map()
h30_daily = h30.copy()
h30_daily["daily_sl"] = h30_daily.index.date
h30_daily["daily_sl"] = h30_daily["daily_sl"].map(sl_map)
h30_daily = h30_daily.dropna(subset=["daily_sl"])

print(f"  Bars after NaN drop: {len(h30_daily)}")
print(f"  Range: {h30_daily.index[0]} — {h30_daily.index[-1]}")

bt_b = Backtest(h30_daily, BasisAdxDailySL, cash=100_000, commission=0.0001, finalize_trades=True)
stats_b = bt_b.run()
rpt_b = BASE / "reports" / "EURUSD_30m_daily_sl.html"
bt_b.plot(filename=str(rpt_b), open_browser=False)
print(f"  Return : {stats_b['Return [%]']:.2f}%")
print(f"  Sharpe : {stats_b['Sharpe Ratio']:.2f}")
print(f"  Max DD : {stats_b['Max. Drawdown [%]']:.2f}%")
print(f"  Trades : {stats_b['# Trades']}")
print(f"  Win%   : {stats_b.get('Win Rate [%]', 0):.1f}%")
print(f"  Best   : {stats_b.get('Best Trade [%]', 0):.2f}%")
print(f"  Worst  : {stats_b.get('Worst Trade [%]', 0):.2f}%")
bt_rpt_b = BASE / "reports" / "EURUSD_30m_daily_sl.html"

# ── SUMMARY ──
print("\n" + "=" * 55)
print("  COMPARISON: 30min EURUSD")
print("=" * 55)
print(f"  {'Metrik':<15} {'SL Same-TF':>12} {'SL Daily':>12}")
print(f"  {'─'*14} {'─'*12} {'─'*12}")
print(f"  {'Return':<15} {stats_a['Return [%]']:>10.2f}% {stats_b['Return [%]']:>10.2f}%")
print(f"  {'Sharpe':<15} {stats_a['Sharpe Ratio']:>11.2f} {stats_b['Sharpe Ratio']:>11.2f}")
print(f"  {'Max DD':<15} {stats_a['Max. Drawdown [%]']:>10.2f}% {stats_b['Max. Drawdown [%]']:>10.2f}%")
print(f"  {'Trades':<15} {stats_a['# Trades']:>11} {stats_b['# Trades']:>11}")
print(f"  {'Win%':<15} {stats_a.get('Win Rate [%]', 0):>10.1f}% {stats_b.get('Win Rate [%]', 0):>10.1f}%")
print(f"  {'Best Trade':<15} {stats_a.get('Best Trade [%]', 0):>10.2f}% {stats_b.get('Best Trade [%]', 0):>10.2f}%")
print(f"  {'Worst Trade':<15} {stats_a.get('Worst Trade [%]', 0):>10.2f}% {stats_b.get('Worst Trade [%]', 0):>10.2f}%")
print()
winner = "SL Same-TF" if stats_a['Return [%]'] > stats_b['Return [%]'] else "SL Daily"
print(f"  🏆 Winner: {winner}")
print(f"\n  Report A: {rpt_a}")
print(f"  Report B: {rpt_b}")
print("=" * 55)
print("✅ DONE")
