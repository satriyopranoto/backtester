"""Middle BB (Basis) + ADX Strategy.

Entry (all must be true on same bar):
  - Low > Donchian SL
  - Close > Basis (middle BB = SMA)
  - ADX > 25
  - +DI (PDI) > -DI (MDI)
  - PDI > PDI[5]  (rising PDI)

Exit:
  - High < current Donchian SL (trailing stop — no fixed entry_sl)

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
    * ADX > 25
    * +DI > -DI
    * +DI > +DI[5]  (PDI rising compared to 5 bars ago)

    Exit
    ----
    * **Trailing stop**: High < current Donchian SL.
      No fixed entry_sl — SL recalculates every bar.
    """

    # ---- tunable parameters ----
    bb_period = 20                # SMA period for basis (middle BB)
    adx_period = 14
    sl_multiple = 2.8
    sl_period = 10
    risk_pct = 1.0                # % equity risked per trade

    # ---- entry-state ----
    _entry_price: float | None = None

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

        # ── ADX / PDI / MDI (pre-computed → len-indexing) ─
        self.adx_arr, self.pdi_arr, self.mdi_arr = calc_adx(
            np.asarray(self.data.High),
            np.asarray(self.data.Low),
            np.asarray(self.data.Close),
            self.adx_period,
        )
        self.I(lambda: self.adx_arr, name="ADX", overlay=False)
        self.I(lambda: self.pdi_arr, name="+DI", overlay=False)
        self.I(lambda: self.mdi_arr, name="-DI", overlay=False)

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

        pdi_5ago = float(self.pdi_arr[idx - 5]) if idx >= 5 else 0.0
        is_nan = np.isnan(adx) or np.isnan(pdi) or np.isnan(mdi)

        # ──────────────────────────────────────────────
        #  1. EXIT — trailing stop (high < current SL)
        # ──────────────────────────────────────────────
        if self._entry_price is not None:
            if high < sl:
                self.position.close()
                self._entry_price = None
                return
            return

        # ──────────────────────────────────────────────
        #  2. ENTRY — when flat
        # ──────────────────────────────────────────────
        if (
            low > sl
            and close > basis
            and not is_nan
            and adx > 25.0
            and pdi > mdi
            and pdi > pdi_5ago
        ):
            # ── Risk-based position sizing ────────
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
                self._entry_price = close
