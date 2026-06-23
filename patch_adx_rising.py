#!/usr/bin/env python3
"""Add ADX > ADX[5] (ADX rising) condition to backtester strategies and screeners."""

import os, sys, py_compile

# ── PATHS ──────────────────────────────────────
BT = "/mnt/c/Users/satri/code/backtester/strategies"
APP = "/home/satri/stocktrade/app.py"

# ─────────────── 1. bb_adx_strategy.py ───────────────
c1 = open(f"{BT}/bb_adx_strategy.py").read()

old1_1 = "pdi_5ago = float(self.pdi_arr[idx - 5]) if idx >= 5 else 0.0"
new1_1 = "pdi_5ago = float(self.pdi_arr[idx - 5]) if idx >= 5 else 0.0\n        adx_5ago = float(self.adx_arr[idx - 5]) if idx >= 5 else 0.0"
c1 = c1.replace(old1_1, new1_1, 1)
print("✓ bb_adx: adx_5ago variable added" if old1_1 in c1 else "✗ bb_adx: adx_5ago FAILED")

old1_2 = """        if (
            close > upper_bb
            and low > sl
            and not is_nan
            and adx > 25.0
            and pdi > mdi             # +DI above -DI
            and pdi > pdi_5ago        # +DI rising
        ):"""
new1_2 = """        if (
            close > upper_bb
            and low > sl
            and not is_nan
            and adx > 25.0
            and adx > adx_5ago        # ADX rising (genuine new trend)
            and pdi > mdi             # +DI above -DI
            and pdi > pdi_5ago        # +DI rising
        ):"""
c1_was = old1_2 in c1
c1 = c1.replace(old1_2, new1_2, 1)
print("✓ bb_adx: adx > adx_5ago added" if c1_was else "✗ bb_adx: condition FAILED")

open(f"{BT}/bb_adx_strategy.py", 'w').write(c1)
print(f"  → saved {len(c1)} chars")


# ─────────────── 2. basis_adx_strategy.py ───────────────
c2 = open(f"{BT}/basis_adx_strategy.py").read()

old2_1 = "pdi_5ago = float(self.pdi_arr[idx - 5]) if idx >= 5 else 0.0"
new2_1 = "pdi_5ago = float(self.pdi_arr[idx - 5]) if idx >= 5 else 0.0\n        adx_5ago = float(self.adx_arr[idx - 5]) if idx >= 5 else 0.0"
c2 = c2.replace(old2_1, new2_1, 1)
print("✓ basis_adx: adx_5ago variable added" if old2_1 in c2 else "✗ basis_adx: adx_5ago FAILED")

old2_2 = """        if (
            low > sl
            and close > basis
            and not is_nan
            and adx > 25.0
            and pdi > mdi
            and pdi > pdi_5ago
        ):"""
new2_2 = """        if (
            low > sl
            and close > basis
            and not is_nan
            and adx > 25.0
            and adx > adx_5ago        # ADX rising (genuine new trend)
            and pdi > mdi
            and pdi > pdi_5ago
        ):"""
c2_was = old2_2 in c2
c2 = c2.replace(old2_2, new2_2, 1)
print("✓ basis_adx: adx > adx_5ago added" if c2_was else "✗ basis_adx: condition FAILED")

open(f"{BT}/basis_adx_strategy.py", 'w').write(c2)
print(f"  → saved {len(c2)} chars")


# ─────────────── 3. app.py - run_bb_screener ───────────────
c3 = open(APP).read()

old3_1 = "pdi_5ago = float(pdi_series.iloc[-6]) if len(pdi_series) >= 6 else 0\n\n                pdi_rising = last_pdi > pdi_5ago"
new3_1 = "pdi_5ago = float(pdi_series.iloc[-6]) if len(pdi_series) >= 6 else 0\n                adx_5ago = float(adx_series.iloc[-6]) if len(adx_series) >= 6 else 0\n\n                pdi_rising = last_pdi > pdi_5ago\n                adx_rising = last_adx > adx_5ago"
c3_was = old3_1 in c3
c3 = c3.replace(old3_1, new3_1, 1)
print("✓ app bb_run: adx_5ago+adx_rising added" if c3_was else "✗ app bb_run: FAILED")

old3_2 = """                    if (not np.isnan(last_adx) and not np.isnan(last_pdi) and not np.isnan(last_mdi)
                            and pdi_above_mdi and adx_strong and pdi_rising):
                        recommendation = \"BUY\""""
new3_2 = """                    if (not np.isnan(last_adx) and not np.isnan(last_pdi) and not np.isnan(last_mdi)
                            and pdi_above_mdi and adx_strong and pdi_rising and adx_rising):
                        recommendation = \"BUY\""""
c3_was2 = old3_2 in c3
c3 = c3.replace(old3_2, new3_2, 1)
print("✓ app bb_run: adx_rising in BUY added" if c3_was2 else "✗ app bb_run: BUY FAILED")


# ─────────────── 4. app.py - run_basis_adx_screener ───────────────
old4_1 = "pdi_5ago = float(pdi_series.iloc[-6]) if len(pdi_series) >= 6 else 0\n                \n                pdi_rising = last_pdi > pdi_5ago"
new4_1 = "pdi_5ago = float(pdi_series.iloc[-6]) if len(pdi_series) >= 6 else 0\n                adx_5ago = float(adx_series.iloc[-6]) if len(adx_series) >= 6 else 0\n                \n                pdi_rising = last_pdi > pdi_5ago\n                adx_rising = last_adx > adx_5ago"
c4_was = old4_1 in c3
c3 = c3.replace(old4_1, new4_1, 1)
print("✓ app basis_run: adx_5ago+adx_rising added" if c4_was else "✗ app basis_run: FAILED")

old4_2 = """                if last_low > last_sl and last_price > last_basis:
                    if (not is_nan
                            and pdi_above_mdi and adx_strong and pdi_rising):
                        recommendation = \"BUY\""""
new4_2 = """                if last_low > last_sl and last_price > last_basis:
                    if (not is_nan
                            and pdi_above_mdi and adx_strong and pdi_rising and adx_rising):
                        recommendation = \"BUY\""""
c4_was2 = old4_2 in c3
c3 = c3.replace(old4_2, new4_2, 1)
print("✓ app basis_run: adx_rising in BUY added" if c4_was2 else "✗ app basis_run: BUY FAILED")

open(APP, 'w').write(c3)
print(f"  → app.py saved ({len(c3)} chars)")


# ── SYNTAX CHECK ──
try:
    py_compile.compile(APP, doraise=True)
    print("✓ app.py syntax OK")
except py_compile.PyCompileError as e:
    print(f"✗ app.py SYNTAX ERROR: {e}")

print("\n✅ All done!")
