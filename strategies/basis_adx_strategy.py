"""Basis ADX Strategy — 3 variants: Standard, Multi-TF (dominance), Multi-TF Modif (dominance + rising).

Class hierarchy:
  BasisAdxStrategy      → H1 only (standard)
  BasisAdxMultiTf       → + daily +DI > daily -DI (LONG) / daily -DI > daily +DI (SHORT)
  BasisAdxMultiTfModif  → + daily DI dominance + daily DI rising (shift 1)

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


# ═══════════════════════════════════════════════════════════════
# 1. STANDARD — pure H1, no multi-TF
# ═══════════════════════════════════════════════════════════════

class BasisAdxStrategy(Strategy):
    """Standard — H1 Basis ADX, no multi-TF filter.

    Entry (LONG, all must be TRUE on same bar):
      * Low > Donchian SL
      * Close > Basis (SMA20)
      * ADX > 20
      * ADX rising (ADX > ADX[5])
      * +DI > -DI
      * +DI rising (PDI > PDI[5])

    Entry (SHORT, all must be TRUE on same bar):
      * High < Donchian SL
      * Close < Basis
      * ADX > 20, ADX rising
      * -DI > +DI
      * -DI rising (MDI > MDI[5])

    Exit via Cut Loss or Take Profit only — no time-based exit.
    """

    # ---- tunable parameters ----
    bb_period = 20
    adx_period = 14
    sl_multiple = 2.8
    sl_period = 10
    tp_min_profit_pct = 0.2
    risk_pct = 1.0

    # ---- entry-state ----
    _entry_sl: float | None = None
    _entry_price: float | None = None
    _tp_threshold_pct: float | None = None
    _is_short: bool = False

    # ────────────────────────────────────
    #  init()
    # ────────────────────────────────────

    def init(self) -> None:
        """Compute H1 indicators once."""
        # ── Basis (SMA of close) ──
        def _basis(arr, period):
            return pd.Series(arr).rolling(period).mean().values

        self.basis = self.I(
            _basis, self.data.Close, self.bb_period,
            name=f"Basis({self.bb_period})", overlay=True,
        )

        # ── Donchian SL ──
        self.sl = self.I(
            donchian_sl, self.data.High, self.data.Low,
            self.sl_multiple, self.sl_period,
            name="SL (Donchian)", overlay=True,
        )

        # ── H1 ADX / PDI / MDI ──
        self.adx_arr, self.pdi_arr, self.mdi_arr = calc_adx(
            np.asarray(self.data.High),
            np.asarray(self.data.Low),
            np.asarray(self.data.Close),
            self.adx_period,
        )
        self.I(
            lambda: pd.DataFrame({
                'ADX': self.adx_arr, '+DI': self.pdi_arr, '-DI': self.mdi_arr,
            }),
            name="ADX/DI", overlay=False,
        )

    # ────────────────────────────────────
    #  Template hooks (override in subclasses)
    # ────────────────────────────────────

    def _mtf_long_ok(self) -> bool:
        """Override to add multi-TF LONG filter. Return True to allow trade."""
        return True

    def _mtf_short_ok(self) -> bool:
        """Override to add multi-TF SHORT filter. Return True to allow trade."""
        return True

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

        # ── 1. EXIT ──
        if self._entry_sl is not None and self._entry_price is not None:
            if self._is_short:
                # SHORT exit
                if close > self._entry_sl:
                    self.position.close(); self._reset(); return
                tp_th = self._tp_threshold_pct if self._tp_threshold_pct is not None else self.tp_min_profit_pct
                if self._entry_price > 0 and tp_th is not None:
                    floating_pct = ((self._entry_price - close) / self._entry_price) * 100.0
                    if floating_pct > tp_th and close > sl:
                        self.position.close(); self._reset(); return
            else:
                # LONG exit
                if close < self._entry_sl:
                    self.position.close(); self._reset(); return
                tp_th = self._tp_threshold_pct if self._tp_threshold_pct is not None else self.tp_min_profit_pct
                if self._entry_price > 0 and tp_th is not None:
                    floating_pct = ((close - self._entry_price) / self._entry_price) * 100.0
                    if floating_pct > tp_th and close < sl:
                        self.position.close(); self._reset(); return
            return

        # ── 2. ENTRY ──
        # LONG
        if (
            low > sl and close > basis and not is_nan
            and adx > 20.0 and adx > adx_5ago
            and pdi > mdi and pdi > pdi_5ago
            and self._mtf_long_ok()
        ):
            self._enter_long(close, sl)
            return

        # SHORT
        if (
            high < sl and close < basis and not is_nan
            and adx > 20.0 and adx > adx_5ago
            and mdi > pdi and mdi > mdi_5ago
            and self._mtf_short_ok()
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
        stop_dist = abs(sl - close)
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


# ═══════════════════════════════════════════════════════════════
# 2. MULTI-TF — daily DI dominance
#    LONG:  daily +DI > daily -DI
#    SHORT: daily -DI > daily +DI
# ═══════════════════════════════════════════════════════════════

class BasisAdxMultiTf(BasisAdxStrategy):
    """Multi-TF — standard H1 conditions + daily DI dominance filter.

    Adds columns ``daily_pdi`` and ``daily_mdi`` (from ``add_daily_di()``).

    LONG  → H1 conditions + **daily +DI > daily -DI**
    SHORT → H1 conditions + **daily -DI > daily +DI**
    """

    def init(self) -> None:
        super().init()
        self._has_daily_dom = (
            hasattr(self.data, 'daily_pdi') and hasattr(self.data, 'daily_mdi')
        )

    def _daily_pdi(self) -> float:
        return float(self.data.daily_pdi[-1])

    def _daily_mdi(self) -> float:
        return float(self.data.daily_mdi[-1])

    def _mtf_long_ok(self) -> bool:
        if not self._has_daily_dom:
            return True
        dp = self._daily_pdi()
        dm = self._daily_mdi()
        if np.isnan(dp) or np.isnan(dm):
            return True
        return dp > dm

    def _mtf_short_ok(self) -> bool:
        if not self._has_daily_dom:
            return True
        dp = self._daily_pdi()
        dm = self._daily_mdi()
        if np.isnan(dp) or np.isnan(dm):
            return True
        return dm > dp


# ═══════════════════════════════════════════════════════════════
# 3. MULTI-TF MODIF — daily DI dominance + rising
#    LONG:  daily +DI > daily -DI  AND  daily +DI > daily +DI[1]
#    SHORT: daily -DI > daily +DI  AND  daily -DI > daily -DI[1]
# ═══════════════════════════════════════════════════════════════

class BasisAdxMultiTfModif(BasisAdxMultiTf):
    """Multi-TF Modif — standard H1 + daily DI dominance + daily DI rising (shift 1).

    Adds columns ``daily_pdi_1ago`` and ``daily_mdi_1ago``.

    LONG  → H1 + daily +DI > daily -DI + **daily +DI > daily +DI[1]**
    SHORT → H1 + daily -DI > daily +DI + **daily -DI > daily -DI[1]**
    """

    def init(self) -> None:
        super().init()
        self._has_daily_rise = (
            hasattr(self.data, 'daily_pdi_1ago') and hasattr(self.data, 'daily_mdi_1ago')
        )

    def _mtf_long_ok(self) -> bool:
        # Parent checks daily +DI > daily -DI
        if not super()._mtf_long_ok():
            return False
        if not self._has_daily_rise:
            return True
        dp = self._daily_pdi()
        dp1 = float(self.data.daily_pdi_1ago[-1])
        if np.isnan(dp) or np.isnan(dp1):
            return True
        return dp > dp1

    def _mtf_short_ok(self) -> bool:
        if not super()._mtf_short_ok():
            return False
        if not self._has_daily_rise:
            return True
        dm = self._daily_mdi()
        dm1 = float(self.data.daily_mdi_1ago[-1])
        if np.isnan(dm) or np.isnan(dm1):
            return True
        return dm > dm1
