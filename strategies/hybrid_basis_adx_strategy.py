"""Hybrid Strategy — Basis+ADX Entry + Mean Rev ADX Exit.

Entry (Basis+ADX — trend detection):
  - Close > Basis (SMA20)
  - Low > Donchian SL
  - ADX > 20 (trending)
  - ADX > ADX[5]  (ADX rising)
  - +DI > -DI
  - +DI > +DI[5]  (PDI rising)

Exit (Mean Rev ADX — trend exhaustion):
  - Cut Loss: Close < entry_sl
  - Take Profit: floating > 0.4 × stop_dist% AND
      ADX < 25 AND ADX < ADX[5] AND
      PDI < PDI[5] AND MDI > PDI

Sizing: risk-based.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting import Strategy
from strategies.bb_adx_strategy import donchian_sl, calc_adx


class HybridBasisAdxStrategy(Strategy):
    """Basis+ADX entry, Mean Rev ADX exit."""

    bb_period = 20
    adx_period = 14
    sl_multiple = 2.8
    sl_period = 10
    adx_entry_threshold = 20      # for entry (Basis+ADX)
    risk_pct = 1.0

    _entry_sl: float | None = None
    _entry_price: float | None = None
    _tp_threshold_pct: float | None = None

    def init(self) -> None:
        def _sma(arr, period):
            return pd.Series(arr).rolling(period).mean().values
        self.basis = self.I(
            _sma, self.data.Close, self.bb_period,
            name=f"SMA({self.bb_period})", overlay=True,
        )
        self.sl = self.I(
            donchian_sl, self.data.High, self.data.Low,
            self.sl_multiple, self.sl_period,
            name="SL (Donchian)", overlay=True,
        )
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
        basis = float(self.basis[-1])
        sl_val = float(self.sl[-1])

        adx = float(self.adx_arr[idx])
        pdi = float(self.pdi_arr[idx])
        mdi = float(self.mdi_arr[idx])
        adx_5ago = float(self.adx_arr[idx - 5]) if idx >= 5 else 0.0
        pdi_5ago = float(self.pdi_arr[idx - 5]) if idx >= 5 else 0.0
        mdi_5ago = float(self.mdi_arr[idx - 5]) if idx >= 5 else 0.0
        is_nan = np.isnan(adx) or np.isnan(pdi) or np.isnan(mdi)

        # ── EXIT (Mean Rev ADX style) ──
        if self._entry_sl is not None and self._entry_price is not None:
            # Cut Loss
            if close < self._entry_sl:
                self.position.close()
                self._entry_sl = None
                self._entry_price = None
                self._tp_threshold_pct = None
                return

            # Take Profit: 0.4R AND trend exhaustion
            tp_th = self._tp_threshold_pct if self._tp_threshold_pct is not None else 0.2
            if self._entry_price > 0 and tp_th is not None:
                floating_pct = ((close - self._entry_price) / self._entry_price) * 100.0
                if floating_pct > tp_th and not is_nan:
                    if (adx < 25 and adx < adx_5ago
                            and pdi < pdi_5ago
                            and mdi > pdi):
                        self.position.close()
                        self._entry_sl = None
                        self._entry_price = None
                        self._tp_threshold_pct = None
                        return
            return

        # ── ENTRY (Basis+ADX style) ──
        if (
            low > sl_val
            and close > basis
            and not is_nan
            and adx > self.adx_entry_threshold
            and adx > adx_5ago
            and pdi > mdi
            and pdi > pdi_5ago
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
