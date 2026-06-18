"""High Breakout + ADX Confirmation Strategy.

Entry (all must be true):
  - Close > Highest(High, HH_WINDOW)[1]   ←  breakout from highest high
  - Low > SL (Donchian)
  - ADX > 25
  - ADX > ADX[5]
  - PDI > MDI

Exit (hanya via cut loss atau take profit):
  - Cut Loss: Close < entry_sl
  - Take Profit: floating P&L > tp_min_profit_pct AND Close < current SL

Sizing: risk-based (risk_pct % of equity / stop_distance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting import Strategy
from strategies.bb_adx_strategy import donchian_sl, calc_adx


class HighBreakoutAdx(Strategy):
    """Breakout dari Highest High + ADX confirmation."""

    # ---- tunable parameters ----
    hh_window = 40               # lookback untuk highest high
    adx_period = 14
    sl_multiple = 2.8
    sl_period = 10
    tp_min_profit_pct = 0.2      # percent
    risk_pct = 1.0               # % equity risked per trade

    # ---- entry-state ----
    _entry_sl: float | None = None
    _entry_price: float | None = None

    # ────────────────────────────────────
    #  init()
    # ────────────────────────────────────

    def init(self) -> None:
        """Compute indicators once."""

        # ── Highest High ───────────────────────────────
        def _highest(arr, window):
            return pd.Series(arr).rolling(window).max().shift(1).values

        self.highest_high = self.I(
            _highest, self.data.High, self.hh_window,
            name=f"Highest({self.hh_window})", overlay=True,
        )

        # ── SL (Donchian, wrapped) ────────────────────
        self.sl = self.I(
            donchian_sl, self.data.High, self.data.Low,
            self.sl_multiple, self.sl_period,
            name="SL (Donchian)", overlay=True,
        )

        # ── ADX / PDI / MDI (pre-computed → len-indexing) ─
        self.adx_arr, self.pdi_arr, self.mdi_arr = calc_adx(
            np.asarray(self.data.High),
            np.asarray(self.data.Low),
            np.asarray(self.data.Close),
            self.adx_period,
        )
        self.I(lambda: pd.DataFrame({'ADX': self.adx_arr, '+DI': self.pdi_arr, '-DI': self.mdi_arr}), name="ADX/DI", overlay=False)

    # ────────────────────────────────────
    #  next()
    # ────────────────────────────────────

    def next(self) -> None:
        idx = len(self.data) - 1

        close = float(self.data.Close[-1])
        low = float(self.data.Low[-1])
        hh = float(self.highest_high[-1])
        sl = float(self.sl[-1])

        adx = float(self.adx_arr[idx])
        pdi = float(self.pdi_arr[idx])
        mdi = float(self.mdi_arr[idx])
        adx_5ago = float(self.adx_arr[idx - 5]) if idx >= 5 else 0.0
        is_nan = np.isnan(adx) or np.isnan(pdi) or np.isnan(mdi)

        # ── 1. EXIT ────────────────────────────────────
        if self._entry_sl is not None and self._entry_price is not None:
            # Cut Loss
            if close < self._entry_sl:
                self.position.close()
                self._entry_sl = None
                self._entry_price = None
                return

            # Take Profit
            if self._entry_price > 0:
                floating_pct = ((close - self._entry_price) / self._entry_price) * 100.0
                if floating_pct > self.tp_min_profit_pct and close < sl:
                    self.position.close()
                    self._entry_sl = None
                    self._entry_price = None
                    return
            return

        # ── 2. ENTRY ───────────────────────────────────
        if (
            close > hh
            and low > sl
            and not is_nan
            and adx > 25.0
            and adx > adx_5ago
            and pdi > mdi
        ):
            stop_dist = abs(close - sl)
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
                self._entry_sl = sl
                self._entry_price = close
