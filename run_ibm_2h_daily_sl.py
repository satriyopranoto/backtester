"""Basis ADX 2H Entry — Daily Donchian SL (multi-timeframe).

Entry signals on 2H, but Stop Loss uses DAILY Donchian SL (wider).
This gives trades more room to breathe, reducing CL from 2H noise.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from backtesting import Backtest, Strategy
from strategies.bb_adx_strategy import calc_adx
import warnings; warnings.filterwarnings("ignore")

BASE = Path(r"C:\Users\satri\code\backtester")

# ────────────────────────────────────────────────────────────
#  1. Load 1D data → compute daily SL (Donchian)
# ────────────────────────────────────────────────────────────
SL_MULTIPLE = 2.8
SL_PERIOD   = 10
ERO = int(SL_MULTIPLE * SL_PERIOD)  # 28

daily_raw = pd.read_csv(BASE/"IBM_1d.txt", parse_dates=["DateTime"])
daily_raw = daily_raw.rename(columns={"DateTime": "Date"}).set_index("Date").sort_index()

# Donchian SL on daily High/Low
def donchian_sl(high, low, ero):
    s_high = pd.Series(high)
    s_low  = pd.Series(low)
    r_prev = s_high.rolling(ero).max().shift(1).values
    s_prev = s_low.rolling(ero).min().shift(1).values
    r_curr = s_high.rolling(ero).max().values
    s_curr = s_low.rolling(ero).min().values
    ab = np.where(high > r_prev, 1, np.where(low < s_prev, -1, 0))
    ac = pd.Series(ab).replace(0, np.nan).ffill().fillna(0).values
    return np.where(ac == 1, s_curr, r_curr)

daily_sl_arr = donchian_sl(
    daily_raw["High"].values,
    daily_raw["Low"].values,
    ERO,
)
daily_raw["daily_sl"] = daily_sl_arr

# Shift by 1 day to avoid look-ahead bias (use PREVIOUS day's SL)
daily_raw["daily_sl"] = daily_raw["daily_sl"].shift(1)

print(f"📅 Daily data : {len(daily_raw)} bars ({daily_raw.index[0].date()} — {daily_raw.index[-1].date()})")
print(f"   Daily SL computed & shifted (no look-ahead bias)")

# ────────────────────────────────────────────────────────────
#  2. Load 2H data → merge daily SL
# ────────────────────────────────────────────────────────────
h2_raw = pd.read_csv(BASE/"IBM_2h.txt", parse_dates=["DateTime"])
h2_raw = h2_raw.rename(columns={"DateTime": "Date"}).set_index("Date").sort_index()

# Create date-only column for merging (floor to day)
h2_raw["day"] = h2_raw.index.floor("D")

# Merge daily_sl from daily data — use merge_asof for robustness
daily_sl_df = daily_raw[["daily_sl"]].reset_index()
daily_sl_df["Date"] = daily_sl_df["Date"].dt.normalize()  # date only
daily_sl_df = daily_sl_df.drop_duplicates(subset="Date").set_index("Date").sort_index()
daily_sl_df.index = daily_sl_df.index.astype("datetime64[ns]")  # match 2H dtype

h2_sorted = h2_raw.reset_index().sort_values("Date")
# Ensure matching dtypes for merge_asof
h2_sorted["day"] = h2_sorted["day"].astype("datetime64[ns]")
h2_merged = pd.merge_asof(
    h2_sorted,
    daily_sl_df,
    left_on="day",
    right_index=True,
)
h2_merged = h2_merged.set_index("Date").sort_index()

# Copy merged data back
h2_raw = h2_merged.copy()
h2_raw = h2_raw.drop(columns=["day"])

# Drop rows where daily_sl is NaN (first ~29 days + 1 shift)
before = len(h2_raw)
h2_raw = h2_raw.dropna(subset=["daily_sl"])
after = len(h2_raw)
print(f"📊 2H data    : {before} bars → {after} after dropping NaN SL")

h2_raw = h2_raw[h2_raw["Volume"] > 0]

print(f"   2H range   : {h2_raw.index[0]} — {h2_raw.index[-1]}")
print(f"   2H bars    : {len(h2_raw)}")

# ────────────────────────────────────────────────────────────
#  3. Strategy — entry on 2H, SL on daily Donchian
# ────────────────────────────────────────────────────────────

class BasisAdx2HDailySL(Strategy):
    """Basis ADX entry on 2H, Stop Loss on Daily Donchian SL.

    Entry (2H):
      - Low > daily_sl
      - Close > Basis(20)
      - ADX(14) > 20
      - +DI > -DI
      - +DI > +DI[5 bars ago] (rising)

    Exit:
      - Cut Loss: Close < entry_sl (daily SL captured at entry)
      - Take Profit: floating > 0.4 × (entry - daily_sl)%  AND Close < current daily_sl
    """

    bb_period   = 20
    adx_period  = 14
    risk_pct    = 1.0      # % equity risked per trade

    _entry_sl: float | None = None
    _entry_price: float | None = None
    _tp_threshold_pct: float | None = None

    def init(self) -> None:
        # ── Basis (SMA of close) on 2H ──
        def _basis(arr, period):
            return pd.Series(arr).rolling(period).mean().values

        self.basis = self.I(
            _basis, self.data.Close, self.bb_period,
            name=f"Basis({self.bb_period})", overlay=True,
        )

        # ── ADX / PDI / MDI on 2H ──
        self.adx_arr, self.pdi_arr, self.mdi_arr = calc_adx(
            np.asarray(self.data.High),
            np.asarray(self.data.Low),
            np.asarray(self.data.Close),
            self.adx_period,
        )
        self.I(lambda: pd.DataFrame({
            'ADX': self.adx_arr,
            '+DI': self.pdi_arr,
            '-DI': self.mdi_arr,
        }), name="ADX/DI", overlay=False)

    def next(self) -> None:
        idx = len(self.data) - 1

        close  = float(self.data.Close[-1])
        low    = float(self.data.Low[-1])
        basis  = float(self.basis[-1])
        daily_sl = float(self.data.daily_sl[-1])  # from merged column

        adx    = float(self.adx_arr[idx])
        pdi    = float(self.pdi_arr[idx])
        mdi    = float(self.mdi_arr[idx])
        pdi_5ago = float(self.pdi_arr[idx - 5]) if idx >= 5 else 0.0

        is_nan = np.isnan(adx) or np.isnan(pdi) or np.isnan(mdi) or np.isnan(daily_sl)

        # ── EXIT ──
        if self._entry_sl is not None and self._entry_price is not None:
            # a) Cut Loss
            if close < self._entry_sl:
                self.position.close()
                self._entry_sl = None
                self._entry_price = None
                self._tp_threshold_pct = None
                return

            # b) Take Profit: floating > 0.4R  AND  close < current daily_sl
            tp_threshold = self._tp_threshold_pct if self._tp_threshold_pct is not None else 999.0
            if self._entry_price > 0:
                floating_pct = ((close - self._entry_price) / self._entry_price) * 100.0
                if floating_pct > tp_threshold and close < daily_sl:
                    self.position.close()
                    self._entry_sl = None
                    self._entry_price = None
                    self._tp_threshold_pct = None
                    return
            return

        # ── ENTRY ──
        if (
            low > daily_sl
            and close > basis
            and not is_nan
            and adx > 20.0
            and pdi > mdi
            and pdi > pdi_5ago
        ):
            stop_dist = abs(close - daily_sl)
            if stop_dist > 0:
                risk_amount = self.equity * (self.risk_pct / 100.0)
                size = int(risk_amount / stop_dist)
                max_by_cash = int((self.equity * 0.95) / close)
                if max_by_cash >= 1:
                    size = max(1, size)
                size = min(size, max_by_cash)
            else:
                size = 0

            if size > 0:
                self.buy(size=size)
                self._entry_sl = daily_sl
                self._entry_price = close
                # 0.4R threshold: 0.4 × (stop as % of entry)
                self._tp_threshold_pct = (stop_dist / close) * 100.0 * 0.4


# ────────────────────────────────────────────────────────────
#  4. Run backtest
# ────────────────────────────────────────────────────────────
print("\n🚀 Running Basis ADX 2H + Daily SL backtest...")

bt = Backtest(h2_raw, BasisAdx2HDailySL, cash=100_000, commission=0.001, finalize_trades=True)
stats = bt.run()

eq_final = stats['Equity Final [$]']
print(f"\n{'='*60}")
print(f"  📊 BASIS ADX 2H + DAILY SL — IBM")
print(f"{'='*60}")
print(f"  Periode      : {stats['Start'].date()} — {stats['End'].date()}")
print(f"  Return       : {stats['Return [%]']:.2f}%")
print(f"  Buy & Hold   : {stats['Buy & Hold Return [%]']:.2f}%")
print(f"  Equity Final : ${eq_final:,.2f}")
print(f"  Equity Peak  : ${stats['Equity Peak [$]']:,.2f}")
print(f"  Sharpe       : {stats['Sharpe Ratio']:.2f}")
print(f"  Max DD       : {stats['Max. Drawdown [%]']:.2f}%")
print(f"  # Trades     : {stats['# Trades']}")
print(f"  Win Rate     : {stats.get('Win Rate [%]', 0):.1f}%")
print(f"  Best Trade   : {stats.get('Best Trade [%]', 0):.2f}%")
print(f"  Worst Trade  : {stats.get('Worst Trade [%]', 0):.2f}%")
print(f"  Avg Trade    : {stats.get('Avg Trade [%]', 0):.2f}%")

# Last trades
trades = stats.get('_trades', None)
if trades is not None and len(trades) > 0:
    print(f"  ──────────────────────────────────────")
    print(f"  Last 10 trades:")
    for i, t in trades.tail(10).iterrows():
        ret = t.ReturnPct * 100
        emoji = "🔥" if ret >= 0 else "❌"
        label = " cut loss" if ret < 0 else ""
        print(f"  #{i}: {t.EntryTime} → {t.ExitTime}  | {ret:+.2f}%  (${t.PnL:+,.0f}) {emoji}{label}")

# ── Save HTML report ──
report_path = BASE / "reports" / "IBM_basis_adx_2h_daily_sl.html"
bt.plot(filename=str(report_path), open_browser=False)
print(f"\n  📄 Report     : {report_path}")
print(f"{'='*60}")
print("✅ DONE")
