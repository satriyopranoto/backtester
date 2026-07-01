"""Mean Reversion ADX+BB Strategy — modified from Basis+ADX.

Entry (all must be true on same bar):
  - Low > Donchian SL
  - ADX < 25  (low volatility / ranging)
  - ADX < ADX[5]  (ADX declining)
  - +DI (PDI) > -DI (MDI)  (bullish bias)
  - +DI > +DI[5]  (PDI rising)
  - -DI < -DI[5]  (MDI declining)

Exit:
  - Cut Loss: Close < entry_sl (fixed SL at entry)
  - Take Profit: floating > 0.4 × stop_dist% AND
      ADX < 25 AND ADX < ADX[5] AND
      PDI < PDI[5] AND MDI > PDI

Sizing: risk-based (risk_pct % of equity / stop_distance).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting import Strategy
from strategies.bb_adx_strategy import donchian_sl, calc_adx


class MeanRevAdxStrategy(Strategy):
    """Mean Reversion via ADX/DI — entry when volatility is low and bullish bias."""

    # ---- tunable parameters ----
    bb_period = 20
    adx_period = 14
    sl_multiple = 2.8
    sl_period = 10
    adx_threshold = 25           # ADX must be below this
    risk_pct = 1.0

    # ---- entry-state ----
    _entry_sl: float | None = None
    _entry_price: float | None = None
    _tp_threshold_pct: float | None = None

    def init(self) -> None:
        # ── Basis (SMA20) for reference ──
        def _sma(arr, period):
            return pd.Series(arr).rolling(period).mean().values
        self.basis = self.I(
            _sma, self.data.Close, self.bb_period,
            name=f"SMA({self.bb_period})", overlay=True,
        )

        # ── Donchian SL ──
        self.sl = self.I(
            donchian_sl, self.data.High, self.data.Low,
            self.sl_multiple, self.sl_period,
            name="SL (Donchian)", overlay=True,
        )

        # ── ADX / PDI / MDI ──
        self.adx_arr, self.pdi_arr, self.mdi_arr = calc_adx(
            np.asarray(self.data.High),
            np.asarray(self.data.Low),
            np.asarray(self.data.Close),
            self.adx_period,
        )
        self.I(
            lambda: pd.DataFrame(
                {"ADX": self.adx_arr, "+DI": self.pdi_arr, "-DI": self.mdi_arr}
            ),
            name="ADX/DI", overlay=False,
        )

    def next(self) -> None:
        idx = len(self.data) - 1

        close = float(self.data.Close[-1])
        low = float(self.data.Low[-1])
        high = float(self.data.High[-1])
        sl_val = float(self.sl[-1])

        adx = float(self.adx_arr[idx])
        pdi = float(self.pdi_arr[idx])
        mdi = float(self.mdi_arr[idx])
        adx_5ago = float(self.adx_arr[idx - 5]) if idx >= 5 else 0.0
        pdi_5ago = float(self.pdi_arr[idx - 5]) if idx >= 5 else 0.0
        mdi_5ago = float(self.mdi_arr[idx - 5]) if idx >= 5 else 0.0
        is_nan = np.isnan(adx) or np.isnan(pdi) or np.isnan(mdi)

        # ──────────────────────────────────────────────
        #  1. EXIT — when in a position
        # ──────────────────────────────────────────────
        if self._entry_sl is not None and self._entry_price is not None:
            # a) Cut Loss
            if low < self._entry_sl:
                self.position.close()
                self._entry_sl = None
                self._entry_price = None
                self._tp_threshold_pct = None
                return

            # b) Take Profit
            tp_threshold = self._tp_threshold_pct if self._tp_threshold_pct is not None else 0.2
            if self._entry_price > 0 and tp_threshold is not None:
                floating_pct = ((close - self._entry_price) / self._entry_price) * 100.0
                if floating_pct > tp_threshold:
                    # Check ADX/DI exit conditions
                    if (
                        not is_nan
                        and adx < self.adx_threshold
                        and adx < adx_5ago
                        and pdi < pdi_5ago
                        and mdi > pdi
                    ):
                        self.position.close()
                        self._entry_sl = None
                        self._entry_price = None
                        self._tp_threshold_pct = None
                        return

            # In position → do NOT check entry signals
            return

        # ──────────────────────────────────────────────
        #  2. ENTRY — when flat
        # ──────────────────────────────────────────────
        if (
            low > sl_val
            and not is_nan
            and adx < self.adx_threshold
            and adx < adx_5ago      # ADX declining
            and pdi > mdi
            and pdi > pdi_5ago      # PDI rising
            and mdi < mdi_5ago      # MDI declining
        ):
            stop_dist = abs(close - sl_val)
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
                self._entry_sl = sl_val
                self._entry_price = close
                self._tp_threshold_pct = (stop_dist / close) * 100.0 * 0.4
