"""Middle BB (Basis) + ADX Strategy with optional Multi-TF daily DI confirmation.

Entry (all must be true on same bar):
  - Low > Donchian SL
  - Close > Basis (middle BB = SMA)
  - ADX > 20
  - ADX > ADX[5]  (rising ADX)
  - +DI (PDI) > -DI (MDI)
  - PDI > PDI[5]  (rising PDI)

Multi-TF (optional, if daily_pdi/daily_mdi columns present):
  - BUY:  also requires Daily +DI > Daily +DI[5]
  - SHORT: Close < Basis, MDI > PDI, MDI rising, Daily -DI > Daily -DI[5]

Exit (LONG):
  - Cut Loss: Close < entry_sl (fixed SL at entry)
  - Take Profit: floating > 0.4 x stop_dist% AND Close < current SL

Exit (SHORT):
  - Cut Loss: Close > entry_sl (short stop)
  - Take Profit: floating > 0.4 x stop_dist% AND Close > current SL

Sizing: risk-based (risk_pct % of equity / stop_distance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting import Strategy
from strategies.bb_adx_strategy import donchian_sl, calc_adx


class BasisAdxStrategy(Strategy):
    """Middle Bollinger Band (Basis) + ADX confirmation.

    Entry
    -----
    * Low > Donchian SL
    * Close > Basis (SMA of close over ``bb_period``)
    * ADX > 20
    * ADX rising (ADX > ADX[5])
    * +DI > -DI
    * +DI rising (PDI > PDI[5])

    Multi-TF (when daily_pdi/mdi columns present in data):
    * BUY:  also requires daily_pdi > daily_pdi_5ago
    * SHORT: Close < Basis, MDI > PDI, MDI rising,
             daily_mdi > daily_mdi_5ago

    Exit (LONG)
    ----
    * **Cut Loss**: Close < entry_sl (fixed SL captured at entry time)
    * **Take Profit**: floating P&L > 0.4 x stop_dist% **AND**
      Close < current Donchian SL (trailing).

    Exit (SHORT)
    ----
    * **Cut Loss**: Close > entry_sl
    * **Take Profit**: floating P&L > 0.4 x stop_dist% **AND**
      Close > current Donchian SL (trailing).
    """

    # ---- tunable parameters ----
    bb_period = 20                # SMA period for basis (middle BB)
    adx_period = 14
    sl_multiple = 2.8
    sl_period = 10
    tp_min_profit_pct = 0.2      # percent
    risk_pct = 1.0                # % equity risked per trade

    # ---- entry-state (persist across next() calls) ----
    _entry_sl: float | None = None
    _entry_price: float | None = None
    _tp_threshold_pct: float | None = None
    _is_short: bool = False       # True = short position

    # ────────────────────────────────────
    #  init()
    # ────────────────────────────────────

    def init(self) -> None:
        """Compute indicators once."""

        # ── Basis (middle BB = SMA of close) ─────────
        def _basis(arr, period):
            return pd.Series(arr).rolling(period).mean().values

        self.basis = self.I(
            _basis, self.data.Close, self.bb_period,
            name=f"Basis({self.bb_period})", overlay=True,
        )

        # ── SL (Donchian, wrapped → safe [-1] indexing) ──
        self.sl = self.I(
            donchian_sl, self.data.High, self.data.Low,
            self.sl_multiple, self.sl_period,
            name="SL (Donchian)", overlay=True,
        )

        # ── ADX / PDI / MDI (pre-computed → len-indexing)
        self.adx_arr, self.pdi_arr, self.mdi_arr = calc_adx(
            np.asarray(self.data.High),
            np.asarray(self.data.Low),
            np.asarray(self.data.Close),
            self.adx_period,
        )
        self.I(lambda: pd.DataFrame({'ADX': self.adx_arr, '+DI': self.pdi_arr, '-DI': self.mdi_arr}), name="ADX/DI", overlay=False)

        # ── Detect if daily DI columns available (multi-TF) ──
        self._has_daily = hasattr(self.data, 'daily_pdi') and hasattr(self.data, 'daily_mdi') and \
                          hasattr(self.data, 'daily_pdi_5ago') and hasattr(self.data, 'daily_mdi_5ago')

    # ────────────────────────────────────
    #  next()
    # ────────────────────────────────────

    def next(self) -> None:
        idx = len(self.data) - 1

        close = float(self.data.Close[-1])
        low = float(self.data.Low[-1])
        high = float(self.data.High[-1])
        basis = float(self.basis[-1])
        sl = float(self.sl[-1])

        adx = float(self.adx_arr[idx])
        pdi = float(self.pdi_arr[idx])
        mdi = float(self.mdi_arr[idx])
        adx_5ago = float(self.adx_arr[idx - 5]) if idx >= 5 else 0.0

        pdi_5ago = float(self.pdi_arr[idx - 5]) if idx >= 5 else 0.0
        mdi_5ago = float(self.mdi_arr[idx - 5]) if idx >= 5 else 0.0
        is_nan = np.isnan(adx) or np.isnan(pdi) or np.isnan(mdi)

        # ── Multi-TF daily DI values ──
        daily_pdi_ok = True
        daily_mdi_ok = True
        if self._has_daily:
            dpdi = float(self.data.daily_pdi[-1])
            dpdi5 = float(self.data.daily_pdi_5ago[-1])
            dmdi = float(self.data.daily_mdi[-1])
            dmdi5 = float(self.data.daily_mdi_5ago[-1])
            daily_pdi_ok = not np.isnan(dpdi) and not np.isnan(dpdi5) and dpdi > dpdi5
            daily_mdi_ok = not np.isnan(dmdi) and not np.isnan(dmdi5) and dmdi > dmdi5

        # ──────────────────────────────────────────────
        #  1. EXIT — when in a position
        # ──────────────────────────────────────────────
        if self._entry_sl is not None and self._entry_price is not None:
            if self._is_short:
                # ── SHORT exit ──
                # a) Cut Loss: price broke above SL (resistance)
                if close > self._entry_sl:
                    self.position.close()
                    self._reset()
                    return

                # b) Take Profit: floating > threshold AND close > current SL (pulled back)
                tp_threshold = self._tp_threshold_pct if self._tp_threshold_pct is not None else self.tp_min_profit_pct
                if self._entry_price > 0 and tp_threshold is not None:
                    # For shorts, profit = entry_price - close (price went down)
                    floating_pct = ((self._entry_price - close) / self._entry_price) * 100.0
                    if floating_pct > tp_threshold and close > sl:
                        self.position.close()
                        self._reset()
                        return
            else:
                # ── LONG exit ──
                # a) Cut Loss
                if close < self._entry_sl:
                    self.position.close()
                    self._reset()
                    return

                # b) Take Profit: floating > dynamic TP threshold AND close < current SL
                tp_threshold = self._tp_threshold_pct if self._tp_threshold_pct is not None else self.tp_min_profit_pct
                if self._entry_price > 0 and tp_threshold is not None:
                    floating_pct = ((close - self._entry_price) / self._entry_price) * 100.0
                    if floating_pct > tp_threshold and close < sl:
                        self.position.close()
                        self._reset()
                        return

            # In position → do NOT check entry signals
            return

        # ──────────────────────────────────────────────
        #  2. ENTRY — when flat
        # ──────────────────────────────────────────────

        # ── LONG entry ──
        if (
            low > sl
            and close > basis
            and not is_nan
            and adx > 20.0 and adx > adx_5ago  # ADX rising
            and pdi > mdi
            and pdi > pdi_5ago
            and daily_pdi_ok                         # multi-TF: daily +DI rising
        ):
            self._enter_long(close, sl)
            return

        # ── SHORT entry ──
        if (
            high < sl                                  # price below SL (resistance)
            and close < basis                          # Close below Basis
            and not is_nan
            and adx > 20.0 and adx > adx_5ago          # ADX rising
            and mdi > pdi                              # -DI > +DI
            and mdi > mdi_5ago                         # -DI rising
            and daily_mdi_ok                           # multi-TF: daily -DI rising
        ):
            self._enter_short(close, sl)
            return

    # ────────────────────────────────────
    #  Helpers
    # ────────────────────────────────────

    def _enter_long(self, close: float, sl: float) -> None:
        stop_dist = abs(close - sl)
        if stop_dist <= 0:
            return
        risk_amount = self.equity * (self.risk_pct / 100.0)
        size = int(risk_amount / stop_dist)
        max_by_cash = int((self.equity * 0.95) / close)
        if max_by_cash >= 1:
            size = max(1, size)
        size = min(size, max_by_cash)
        if size <= 0:
            return

        self.buy(size=size)
        self._entry_sl = sl
        self._entry_price = close
        self._tp_threshold_pct = (stop_dist / close) * 100.0 * 0.4
        self._is_short = False

    def _enter_short(self, close: float, sl: float) -> None:
        stop_dist = abs(sl - close)  # distance for short
        if stop_dist <= 0:
            return
        risk_amount = self.equity * (self.risk_pct / 100.0)
        size = int(risk_amount / stop_dist)
        max_by_cash = int((self.equity * 0.95) / close)
        if max_by_cash >= 1:
            size = max(1, size)
        size = min(size, max_by_cash)
        if size <= 0:
            return

        self.sell(size=size)
        self._entry_sl = sl
        self._entry_price = close
        self._tp_threshold_pct = (stop_dist / close) * 100.0 * 0.4
        self._is_short = True

    def _reset(self) -> None:
        self._entry_sl = None
        self._entry_price = None
        self._tp_threshold_pct = None
        self._is_short = False
