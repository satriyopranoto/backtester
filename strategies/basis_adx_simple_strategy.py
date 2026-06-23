"""Basis ADX baseline (ADX > 20 + ADX rising, no SMA200, 0.4R TP)."""
import numpy as np
import pandas as pd
from backtesting import Strategy
from strategies.bb_adx_strategy import donchian_sl, calc_adx


class BasisAdxSimple(Strategy):
    """Middle BB + ADX > 20 + ADX rising + PDI rising (BASELINE)."""

    bb_period = 20
    adx_period = 14
    sl_multiple = 2.8
    sl_period = 10
    tp_min_profit_pct = 0.2
    risk_pct = 1.0

    _entry_sl: float | None = None
    _entry_price: float | None = None
    _tp_threshold_pct: float | None = None

    def init(self) -> None:
        def _basis(arr, period):
            return pd.Series(arr).rolling(period).mean().values

        self.basis = self.I(
            _basis, self.data.Close, self.bb_period,
            name=f"Basis({self.bb_period})", overlay=True,
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
        self.I(lambda: pd.DataFrame({'ADX': self.adx_arr, '+DI': self.pdi_arr, '-DI': self.mdi_arr}), name="ADX/DI", overlay=False)

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
        adx_5ago = float(self.adx_arr[idx - 5]) if idx >= 5 else 0.0  # ADX rising
        is_nan = np.isnan(adx) or np.isnan(pdi) or np.isnan(mdi)

        # ── EXIT ──
        if self._entry_sl is not None and self._entry_price is not None:
            if close < self._entry_sl:
                self.position.close()
                self._entry_sl = None
                self._entry_price = None
                self._tp_threshold_pct = None
                return

            tp_threshold = self._tp_threshold_pct if self._tp_threshold_pct is not None else self.tp_min_profit_pct
            if self._entry_price > 0 and tp_threshold is not None:
                floating_pct = ((close - self._entry_price) / self._entry_price) * 100.0
                if floating_pct > tp_threshold and close < sl:
                    self.position.close()
                    self._entry_sl = None
                    self._entry_price = None
                    self._tp_threshold_pct = None
                    return
            return

        # ── ENTRY (BASELINE) ──
        if (
            low > sl
            and close > basis
            and not is_nan
            and adx > 20.0
            and adx > adx_5ago          # ADX rising
            and pdi > mdi
            and pdi > pdi_5ago
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
                self._tp_threshold_pct = (stop_dist / close) * 100.0 * 0.4
