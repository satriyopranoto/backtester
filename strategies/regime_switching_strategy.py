"""Regime-Switching Strategy — Trend vs Mean Reversion.

Regime detection (last 100 bars):
  - trend_score >= MIN_TREND_SCORE  →  TREND MODE (Basis+ADX)
  - trend_score <  MIN_TREND_SCORE  →  MEAN REVERSION MODE

Trend Mode (Basis+ADX):
  Entry: ADX>25, +DI>-DI, +DI rising, ADX rising, Close>Basis, Low>SL
  Exit : SL / TP 0.4R trailing

Mean Reversion Mode:
  Entry: RSI < oversold AND Low < Lower BB
  Exit : RSI > overbought AND High > Upper BB  OR  SL / TP
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting import Strategy
from strategies.bb_adx_strategy import donchian_sl, calc_adx


class RegimeSwitchingStrategy(Strategy):
    """Switches between Trend (Basis+ADX) and Mean Reversion based on market regime."""

    # ---- shared params ----
    bb_period = 20
    bb_std = 2.0
    adx_period = 14
    sl_multiple = 2.8
    sl_period = 10
    risk_pct = 1.0

    # ---- regime params ----
    trend_bars = 100
    min_trend_score = 30          # % : >= this → trend mode, < this → mean rev
    adx_threshold = 25

    # ---- mean reversion params ----
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70

    # ---- entry-state ----
    _entry_sl: float | None = None
    _entry_price: float | None = None
    _tp_threshold_pct: float | None = None
    _current_mode: str = "unknown"  # "trend" or "meanrev"

    def init(self) -> None:
        # ── Basis (SMA20) ──
        def _sma(arr, period):
            return pd.Series(arr).rolling(period).mean().values
        self.basis = self.I(
            _sma, self.data.Close, self.bb_period,
            name=f"SMA({self.bb_period})", overlay=True,
        )

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

        # ── RSI ──
        def _rsi(arr, period):
            s = pd.Series(arr)
            delta = s.diff()
            gain = delta.clip(lower=0).ewm(alpha=1/period).mean()
            loss = (-delta).clip(lower=0).ewm(alpha=1/period).mean()
            rs = gain / loss.replace(0, np.nan)
            return (100 - (100 / (1 + rs))).values
        self.rsi = self.I(_rsi, self.data.Close, self.rsi_period, name="RSI", overlay=False)

    def _trend_score(self, idx: int) -> float:
        """% of last N bars where ADX>25 AND Close>SMA20."""
        if idx < self.trend_bars:
            return 0.0
        start = idx - self.trend_bars + 1
        adx_slice = self.adx_arr[start:idx + 1]
        close_slice = np.asarray(self.data.Close)[start:idx + 1]
        sma_slice = np.asarray(self.basis)[start:idx + 1]
        n_adx = np.sum(adx_slice > self.adx_threshold)
        n_bull = np.sum(close_slice > sma_slice)
        both = sum(1 for i in range(len(adx_slice))
                   if adx_slice[i] > self.adx_threshold and close_slice[i] > sma_slice[i])
        return (both / self.trend_bars) * 100.0

    def next(self) -> None:
        idx = len(self.data) - 1
        close = float(self.data.Close[-1])
        low = float(self.data.Low[-1])
        high = float(self.data.High[-1])
        basis = float(self.basis[-1])
        sl_val = float(self.sl[-1])
        lower_bb = float(self.lower_bb[-1])
        upper_bb = float(self.upper_bb[-1])
        rsi = float(self.rsi[-1])

        adx = float(self.adx_arr[idx])
        pdi = float(self.pdi_arr[idx])
        mdi = float(self.mdi_arr[idx])
        adx_5ago = float(self.adx_arr[idx - 5]) if idx >= 5 else 0.0
        pdi_5ago = float(self.pdi_arr[idx - 5]) if idx >= 5 else 0.0

        # ── Regime detection ──
        score = self._trend_score(idx)
        in_trend_mode = score >= self.min_trend_score
        self._current_mode = "trend" if in_trend_mode else "meanrev"

        # ──────────────────────────────────────────────
        #  1. EXIT
        # ──────────────────────────────────────────────
        if self._entry_sl is not None and self._entry_price is not None:
            # a) Cut Loss (always)
            if close < self._entry_sl:
                self.position.close()
                self._entry_sl = self._entry_price = self._tp_threshold_pct = None
                return

            # b) TP trailing
            tp_th = self._tp_threshold_pct if self._tp_threshold_pct is not None else 0.2
            if self._entry_price > 0 and tp_th is not None:
                floating = ((close - self._entry_price) / self._entry_price) * 100.0
                if floating > tp_th and close < sl_val:
                    self.position.close()
                    self._entry_sl = self._entry_price = self._tp_threshold_pct = None
                    return

            # c) Mean reversion exit: RSI overbought + price above upper BB
            if not in_trend_mode:
                if rsi > self.rsi_overbought and high > upper_bb:
                    self.position.close()
                    self._entry_sl = self._entry_price = self._tp_threshold_pct = None
                    return

            # In position → no new entry
            return

        # ──────────────────────────────────────────────
        #  2. ENTRY
        # ──────────────────────────────────────────────
        if in_trend_mode:
            # ── TREND MODE: Basis+ADX ──
            is_nan = np.isnan(adx) or np.isnan(pdi) or np.isnan(mdi)
            if (
                low > sl_val
                and close > basis
                and not is_nan
                and adx > self.adx_threshold
                and adx > adx_5ago
                and pdi > mdi
                and pdi > pdi_5ago
            ):
                stop_dist = abs(close - sl_val)
                if stop_dist > 0:
                    risk = self.equity * (self.risk_pct / 100.0)
                    size = max(1, int(risk / stop_dist))
                    max_cash = int((self.equity * 0.95) / close)
                    if max_cash > 0:
                        size = min(size, max_cash)
                    self.buy(size=size)
                    self._entry_sl = sl_val
                    self._entry_price = close
                    self._tp_threshold_pct = (stop_dist / close) * 100.0 * 0.4
        else:
            # ── MEAN REVERSION MODE ──
            if (
                low < lower_bb
                and not np.isnan(rsi)
                and rsi < self.rsi_oversold
            ):
                stop_dist = abs(close - sl_val)
                if stop_dist > 0:
                    risk = self.equity * (self.risk_pct / 100.0)
                    size = max(1, int(risk / stop_dist))
                    max_cash = int((self.equity * 0.95) / close)
                    if max_cash > 0:
                        size = min(size, max_cash)
                    self.buy(size=size)
                    self._entry_sl = sl_val
                    self._entry_price = close
                    self._tp_threshold_pct = (stop_dist / close) * 100.0 * 0.4
