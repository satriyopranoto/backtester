"""BB Breakout + ADX Confirmation Strategy.

Logic diadaptasi dari stocktrade (app.py):
  - calculate_sl()       -> Donchian Channel SL (ero = 28)
  - calculate_bollinger_bands() -> BB(20, 2)
  - calculate_adx()      -> ADX, +DI (PDI), -DI (MDI) Wilder's smoothing
  - BB Screener buy: close > upper_bb AND price > sl
                     AND adx > 25 AND adx_rising AND pdi > mdi

Entry: simpan entry_sl dan entry_price SEKALI (entry pertama).
       Buy signal berikutnya saat sudah in-position DIIGONTAIN.
Exit : hanya via Cut Loss (close < entry_sl) atau Take Profit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting import Strategy


# ─────────────────────────────────────────────────────────────
#  Pure helper functions (no Backtesting.py coupling)
# ─────────────────────────────────────────────────────────────


def donchian_sl(
    high: np.ndarray,
    low: np.ndarray,
    atr_multiple: float = 2.8,
    atr_period: int = 10,
) -> np.ndarray:
    """Donchian Channel SL — ported from stocktrade ``calculate_sl()``.

    Parameters
    ----------
    high, low : 1-D arrays (prices in original scale).
    atr_multiple, atr_period : ero = int(atr_multiple * atr_period).

    Returns
    -------
    sl : 1-D array, same length as inputs.
    """
    ero = int(atr_multiple * atr_period)

    s_high = pd.Series(high)
    s_low = pd.Series(low)

    r_prev = s_high.rolling(window=ero).max().shift(1).values
    s_prev = s_low.rolling(window=ero).min().shift(1).values
    r_curr = s_high.rolling(window=ero).max().values
    s_curr = s_low.rolling(window=ero).min().values

    # ab: trigger direction
    ab = np.where(high > r_prev, 1, np.where(low < s_prev, -1, 0))

    # ac = valuewhen(ab != 0, ab, 0)  → forward-fill last non-zero
    ac = pd.Series(ab).replace(0, np.nan).ffill().fillna(0).values

    sl = np.where(ac == 1, s_curr, r_curr)
    return sl


def calc_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ADX, +DI (PDI), -DI (MDI) — ported from stocktrade ``calculate_adx()``.

    Returns (adx, pdi, mdi) arrays, same length as inputs.
    """
    s_high = pd.Series(high)
    s_low = pd.Series(low)
    s_close = pd.Series(close)

    prev_close = s_close.shift(1)

    tr = pd.concat(
        [
            s_high - s_low,
            (s_high - prev_close).abs(),
            (s_low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = s_high - s_high.shift(1)
    down_move = s_low.shift(1) - s_low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0),
        index=s_close.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0),
        index=s_close.index,
    )

    alpha = 1.0 / period
    smoothed_tr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    smoothed_plus = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    smoothed_minus = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    pdi = 100 * smoothed_plus / smoothed_tr.replace(0, np.nan)
    mdi = 100 * smoothed_minus / smoothed_tr.replace(0, np.nan)

    dm_sum = pdi + mdi
    dx = 100 * (pdi - mdi).abs() / dm_sum.replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    return adx.values, pdi.values, mdi.values


# ─────────────────────────────────────────────────────────────
#  Strategy class
# ─────────────────────────────────────────────────────────────


class BbAdxStrategy(Strategy):
    """BB Breakout + ADX Confirmation — only exit via cut-loss or take-profit.

    **Entry** (all must be true on the same bar):

    * Close > Upper Bollinger Band
    * Low > Donchian SL
    * ADX > 25
    * +DI > -DI
    * +DI > +DI[5] (+DI *rising*)

    **Exit** (checked every bar while in position):

    * **Cut Loss**: Close < **entry_sl**  (SL value fixed at entry time).
    * **Take Profit**: unrealised P&L > ``tp_min_profit_pct`` **AND**
      Close < *current* Donchian SL (trailing).
    """

    # ---- tunable parameters ----
    bb_period = 20
    bb_std = 2.0
    adx_period = 14
    sl_multiple = 2.8
    sl_period = 10
    tp_min_profit_pct = 0.2  # percent
    risk_pct = 1.0             # % equity risked per trade

    # ---- entry-state (persist across next() calls) ----
    _entry_sl: float | None = None
    _entry_price: float | None = None

    # ────────────────────────────────────
    #  init()
    # ────────────────────────────────────

    def init(self) -> None:
        """Compute all indicators once.  Use **len(self.data) indexing**
        inside ``next()`` for arrays that are **not** wrapped by ``self.I()``
        to avoid look-ahead bias.
        """
        # ── BB (wrapped → safe [-1] indexing in next()) ──────────
        def _bb_upper(arr, period, k):
            s = pd.Series(arr)
            m = s.rolling(period).mean()
            return (m + k * s.rolling(period).std()).values

        self.upper_bb = self.I(
            _bb_upper, self.data.Close,
            self.bb_period, self.bb_std,
            name="Upper BB", overlay=True,
        )

        self.lower_bb = self.I(
            lambda arr, p, k: (
                (m := pd.Series(arr).rolling(p).mean())
                - k * pd.Series(arr).rolling(p).std()
            ).values,
            self.data.Close, self.bb_period, self.bb_std,
            name="Lower BB", overlay=True,
        )

        # ── SL (wrapped → safe [-1] indexing) ──────────────────
        self.sl = self.I(
            donchian_sl, self.data.High, self.data.Low,
            self.sl_multiple, self.sl_period,
            name="SL (Donchian)", overlay=True,
        )

        # ── ADX / PDI / MDI (pre-computed → use len-indexing!) ─
        self.adx_arr, self.pdi_arr, self.mdi_arr = calc_adx(
            np.asarray(self.data.High),
            np.asarray(self.data.Low),
            np.asarray(self.data.Close),
            self.adx_period,
        )

        # Register for plot only — combined ADX/DI panel
        self.I(lambda: pd.DataFrame({'ADX': self.adx_arr, '+DI': self.pdi_arr, '-DI': self.mdi_arr}), name="ADX/DI", overlay=False)

    # ────────────────────────────────────
    #  next()
    # ────────────────────────────────────

    def next(self) -> None:
        """Per-bar logic."""
        idx = len(self.data) - 1  # current bar index

        # ── Unwrap current values ──────────────────────────────
        close = float(self.data.Close[-1])
        low = float(self.data.Low[-1])
        upper_bb = float(self.upper_bb[-1])
        sl = float(self.sl[-1])

        adx = float(self.adx_arr[idx])
        pdi = float(self.pdi_arr[idx])
        mdi = float(self.mdi_arr[idx])
        pdi_5ago = float(self.pdi_arr[idx - 5]) if idx >= 5 else 0.0
        adx_5ago = float(self.adx_arr[idx - 5]) if idx >= 5 else 0.0

        is_nan = np.isnan(adx) or np.isnan(pdi) or np.isnan(mdi)

        # ──────────────────────────────────────────────────────
        #  1. EXIT — when in a position
        # ──────────────────────────────────────────────────────
        if self._entry_sl is not None and self._entry_price is not None:
            # a) Cut Loss
            if close < self._entry_sl:
                self.position.close()
                self._entry_sl = None
                self._entry_price = None
                return

            # b) Take Profit: floating > min %  AND  close < current SL
            if self._entry_price > 0:
                floating_pct = ((close - self._entry_price) / self._entry_price) * 100.0
                if floating_pct > self.tp_min_profit_pct and close < sl:
                    self.position.close()
                    self._entry_sl = None
                    self._entry_price = None
                    return

            # In position → do NOT check entry signals
            return

        # ──────────────────────────────────────────────────────
        #  2. ENTRY — only when flat
        # ──────────────────────────────────────────────────────
        if (
            close > upper_bb
            and low > sl
            and not is_nan
            and adx > 25.0
            and adx > adx_5ago        # ADX rising (genuine new trend)
            and pdi > mdi             # +DI above -DI
            and pdi > pdi_5ago        # +DI rising
        ):
            # ── Risk-based position sizing ────────────────
            # risk_amount  = equity × (risk_pct / 100)
            # stop_dist    = entry_price - stop_loss (entry_sl)
            # size         = risk_amount / stop_dist
            stop_dist = abs(close - sl)
            if stop_dist > 0:
                risk_amount = self.equity * (self.risk_pct / 100.0)
                size = int(risk_amount / stop_dist)
                # Minimum 1 unit during affordable
                max_by_cash = int((self.equity * 0.95) / close)
                if max_by_cash >= 1:
                    size = max(1, size)
                size = min(size, max_by_cash)
            else:
                size = 0

            if size > 0:
                self.buy(size=size)
                # 🔒 Capture ONCE — never overwritten until position closes
                self._entry_sl = sl
                self._entry_price = close
