"""Basis ADX 1H Entry — Daily Donchian SL (multi-timeframe).

Entry signals on 1H, but Stop Loss uses DAILY Donchian SL (wider).
Daily SL is drawn on the chart HTML.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from backtesting import Backtest, Strategy
from strategies.bb_adx_strategy import calc_adx, donchian_sl
import warnings; warnings.filterwarnings("ignore")

BASE = Path(r"C:\Users\satri\code\backtester")

# ── Params ──
SL_MULTIPLE = 2.8
SL_PERIOD   = 10
ERO = int(SL_MULTIPLE * SL_PERIOD)  # 28

# ────────────────────────────────────────────────────────────
#  1. Load daily data → compute daily SL
# ────────────────────────────────────────────────────────────
print("=" * 55)
print("  EURUSD — Daily SL Computation")
print("=" * 55)

daily_raw = pd.read_csv(BASE / "EURUSD_1d_yf.txt")
daily_raw["Date"] = pd.to_datetime(daily_raw["Date"])
daily_raw = daily_raw.set_index("Date").sort_index()

# Donchian SL on daily bars
daily_sl_arr = donchian_sl(
    daily_raw["High"].values,
    daily_raw["Low"].values,
    SL_MULTIPLE,
    SL_PERIOD,
)
daily_raw["daily_sl"] = daily_sl_arr
daily_raw["daily_sl"] = daily_raw["daily_sl"].shift(1)  # no look-ahead

print(f"  Daily bars : {len(daily_raw)} ({daily_raw.index[0].date()} — {daily_raw.index[-1].date()})")
print(f"  Daily SL non-NaN: {daily_raw['daily_sl'].notna().sum()} of {len(daily_raw)}")

# Build date → SL map
sl_map = daily_raw["daily_sl"].to_dict()
sl_map = {k.date(): v for k, v in sl_map.items() if not np.isnan(v) if pd.notna(v)}

# ────────────────────────────────────────────────────────────
#  2. Load 1H data → merge daily SL
# ────────────────────────────────────────────────────────────
h1_raw = pd.read_csv(BASE / "EURUSD_1h_yf.txt")
h1_raw["Datetime"] = pd.to_datetime(h1_raw["Datetime"], utc=True)
h1_raw = h1_raw.rename(columns={"Datetime": "Date"}).set_index("Date").sort_index()
h1_raw.index = h1_raw.index.tz_localize(None)

# Map daily_sl using date part
h1_raw["daily_sl"] = h1_raw.index.date
h1_raw["daily_sl"] = h1_raw["daily_sl"].map(sl_map)

# Drop NaN daily_sl
before = len(h1_raw)
h1_raw = h1_raw.dropna(subset=["daily_sl"])
print(f"  1H bars    : {before} → {len(h1_raw)} after dropping NaN daily_sl")
print(f"  1H range   : {h1_raw.index[0]} — {h1_raw.index[-1]}")

# Keep only OHLCV + daily_sl
keep = ["Open", "High", "Low", "Close", "Volume", "daily_sl"]
h1_raw = h1_raw[[c for c in keep if c in h1_raw.columns]]

print(f"  1H bars    : {len(h1_raw)}")

# ────────────────────────────────────────────────────────────
#  3. Strategy — entry on 1H, SL on daily Donchian
# ────────────────────────────────────────────────────────────

class BasisAdx1HDailySL(Strategy):
    """Basis ADX entry on 1H, Stop Loss on Daily Donchian SL (drawn on chart).

    Entry (1H):
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
        # ── Basis (SMA of close) on 1H ──
        def _basis(arr, period):
            return pd.Series(arr).rolling(period).mean().values

        self.basis = self.I(
            _basis, self.data.Close, self.bb_period,
            name=f"Basis({self.bb_period})", overlay=True,
        )

        # ── Daily SL line on chart ──
        self.daily_sl_line = self.I(
            lambda: self.data.daily_sl,
            name="Daily SL (Donchian)", overlay=True,
            color="red",
        )

        # ── ADX / PDI / MDI on 1H ──
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

        close    = float(self.data.Close[-1])
        low      = float(self.data.Low[-1])
        basis    = float(self.basis[-1])
        daily_sl = float(self.data.daily_sl[-1])

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
print("\n" + "=" * 55)
print("  🚀 Basis ADX 1H + Daily SL — EURUSD")
print("=" * 55)

bt = Backtest(h1_raw, BasisAdx1HDailySL, cash=100_000, commission=0.0001, finalize_trades=True)
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

# ── Save HTML report ──
report_path = BASE / "reports" / "EURUSD_basis_adx_1h_daily_sl.html"
bt.plot(filename=str(report_path), open_browser=False)
print(f"\n  📄 Report     : {report_path}")
print("=" * 55)
print("✅ DONE")
