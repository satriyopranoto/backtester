"""Mean Reversion Strategy — RSI + Bollinger Bands.

Entry (BUY):
  - RSI < 20  (oversold)
  - Low < Lower Bollinger Band

Exit (SELL):
  - RSI > 70  (overbought)
  - High > Upper Bollinger Band

Also includes SL (Donchian) and TP (0.4R trailing) for risk management.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting import Strategy
from strategies.bb_adx_strategy import donchian_sl


class MeanReversionStrategy(Strategy):
    """Mean Reversion — RSI oversold bounce, overbought exit."""

    # ---- tunable parameters ----
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    rsi_oversold = 20
    rsi_overbought = 70
    sl_multiple = 2.8
    sl_period = 10
    risk_pct = 1.0                # % equity risked per trade

    # ---- entry-state ----
    _entry_sl: float | None = None
    _entry_price: float | None = None
    _tp_threshold_pct: float | None = None

    def init(self) -> None:
        # ── Bollinger Bands ──
        def _bb(arr, period, k):
            s = pd.Series(arr)
            m = s.rolling(period).mean()
            std = s.rolling(period).std()
            return m.values, (m + k * std).values, (m - k * std).values

        self.sma, self.upper_bb, self.lower_bb = self.I(
            _bb, self.data.Close, self.bb_period, self.bb_std,
            name="BB(20,2)", overlay=True,
        )

        # ── Donchian SL ──
        self.sl = self.I(
            donchian_sl, self.data.High, self.data.Low,
            self.sl_multiple, self.sl_period,
            name="SL (Donchian)", overlay=True,
        )

        # ── RSI ──
        def _rsi(arr, period):
            s = pd.Series(arr)
            delta = s.diff()
            gain = delta.clip(lower=0).ewm(alpha=1/period).mean()
            loss = (-delta).clip(lower=0).ewm(alpha=1/period).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            return rsi.values

        self.rsi = self.I(_rsi, self.data.Close, self.rsi_period, name="RSI", overlay=False)

    def next(self) -> None:
        close = float(self.data.Close[-1])
        low = float(self.data.Low[-1])
        high = float(self.data.High[-1])
        lower_bb = float(self.lower_bb[-1])
        upper_bb = float(self.upper_bb[-1])
        sl_val = float(self.sl[-1])
        rsi = float(self.rsi[-1])

        # ──────────────────────────────────────────────
        #  1. EXIT
        # ──────────────────────────────────────────────
        if self._entry_sl is not None and self._entry_price is not None:
            # a) Cut Loss (always)
            if close < self._entry_sl:
                self.position.close()
                self._entry_sl = None
                self._entry_price = None
                self._tp_threshold_pct = None
                return

            # b) Take Profit: floating > TP threshold  AND  close < current SL
            tp_threshold = self._tp_threshold_pct if self._tp_threshold_pct is not None else 0.2
            if self._entry_price > 0 and tp_threshold is not None:
                floating_pct = ((close - self._entry_price) / self._entry_price) * 100.0
                if floating_pct > tp_threshold and close < sl_val:
                    self.position.close()
                    self._entry_sl = None
                    self._entry_price = None
                    self._tp_threshold_pct = None
                    return

            # c) RSI overbought exit
            if rsi > self.rsi_overbought and high > upper_bb:
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
            low < lower_bb                       # Price below lower BB
            and not np.isnan(rsi)
            and rsi < self.rsi_oversold           # RSI oversold
        ):
            # ── Risk-based sizing ──
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
