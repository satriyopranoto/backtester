"""Scan multiple assets for Basis ADX compatibility."""
import yfinance as yf
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, r"C:\Users\satri\code\backtester")
from strategies.bb_adx_strategy import calc_adx


def scan_adx(ticker, interval, period, label=""):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
        if df.empty or len(df) < 50:
            return None
        if df.columns.nlevels > 1:
            df.columns = [c[0] for c in df.columns]
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        c = df["Close"].values.astype(float)
        h = df["High"].values.astype(float)
        l = df["Low"].values.astype(float)

        adx, pdi, mdi = calc_adx(h, l, c, 14)
        valid = adx[~np.isnan(adx)]
        if len(valid) < 10:
            return None

        avg_adx = float(np.mean(valid))
        pct_gt20 = float(np.mean(valid > 20) * 100)

        pdi_valid = pdi[~np.isnan(pdi) & ~np.isnan(mdi)]
        mdi_valid = mdi[~np.isnan(pdi) & ~np.isnan(mdi)]
        pdi_bias = float(np.mean(pdi_valid > mdi_valid) * 100) if len(pdi_valid) > 0 else 0.0

        entry_count = 0
        for i in range(5, len(adx)):
            if (not np.isnan(adx[i]) and not np.isnan(pdi[i])
                    and not np.isnan(mdi[i]) and not np.isnan(pdi[i - 5])):
                if adx[i] > 20 and pdi[i] > mdi[i] and pdi[i] > pdi[i - 5]:
                    entry_count += 1
        entry_freq = float(entry_count / max(len(adx), 1) * 100)

        return {"avg_adx": avg_adx, "adx_gt20": pct_gt20, "pdi_bias": pdi_bias,
                "entry_freq": entry_freq, "bars": len(df), "label": label or ticker}
    except Exception:
        return None


def score_asset(data):
    if data is None:
        return 0.0
    s = 0.0
    s += min(data["adx_gt20"] / 10.0, 25.0)
    s += min(data["avg_adx"] * 1.5, 20.0)
    s += min(data["pdi_bias"] / 5.0, 15.0)
    s += min(data["entry_freq"] * 25.0, 20.0)
    s += min(data["bars"] / 2000.0 * 10.0, 10.0)
    return min(s, 90.0)


all_assets = [
    ("EURUSD=X", "EURUSD"), ("GBPUSD=X", "GBPUSD"), ("AUDUSD=X", "AUDUSD"),
    ("USDJPY=X", "USDJPY"), ("EURJPY=X", "EURJPY"), ("GBPJPY=X", "GBPJPY"),
    ("NZDUSD=X", "NZDUSD"), ("USDCAD=X", "USDCAD"),
    ("GC=F", "Gold"), ("SI=F", "Silver"), ("CL=F", "Crude Oil"),
    ("NG=F", "Nat Gas"), ("HG=F", "Copper"),
    ("SPY", "S&P500"), ("QQQ", "NASDAQ"), ("AAPL", "Apple"), ("MSFT", "MSFT"),
    ("TSLA", "TSLA"), ("NVDA", "NVDA"),
    ("BTC-USD", "Bitcoin"), ("ETH-USD", "Ethereum"), ("SOL-USD", "Solana"),
    ("BBRI.JK", "BBRI"), ("TLKM.JK", "TLKM"), ("BBCA.JK", "BBCA"),
    ("ASII.JK", "ASII"), ("ADES.JK", "ADES"), ("UNVR.JK", "UNVR"),
]

# ── 1H Scan ──
print("=" * 65)
print("  SCAN 1H (1 YEAR)")
print("=" * 65)
print(f"  {'#':<3} {'Aset':<12} {'ADX':<6} {'>20%':<6} {'Entry%':<7} {'PDI%':<6} {'Bars':<6} {'Score':<5}")
print(f"  {'─'*2} {'─'*11} {'─'*5} {'─'*5} {'─'*6} {'─'*5} {'─'*5} {'─'*4}")

res1 = []
for t, label in all_assets:
    d = scan_adx(t, "1h", "1y", label)
    if d:
        s = score_asset(d)
        res1.append((s, d))

res1.sort(key=lambda x: x[0], reverse=True)

for i, (s, d) in enumerate(res1):
    e = "🔥" if s >= 70 else ("✅" if s >= 55 else ("⚠️" if s >= 40 else "❌"))
    print(f"  {e} {i+1:<2} {d['label']:<12} {d['avg_adx']:<5.1f} {d['adx_gt20']:<4.0f}% {d['entry_freq']:<5.2f}% {d['pdi_bias']:<4.0f}% {d['bars']:<5} {s:<3.0f}")

# ── 4H Scan ──
print(f"\n\n{'='*65}")
print("  SCAN 4H (2 YEARS)")
print("=" * 65)
print(f"  {'#':<3} {'Aset':<12} {'ADX':<6} {'>20%':<6} {'Entry%':<7} {'PDI%':<6} {'Bars':<6} {'Score':<5}")
print(f"  {'─'*2} {'─'*11} {'─'*5} {'─'*5} {'─'*6} {'─'*5} {'─'*5} {'─'*4}")

res4 = []
for t, label in all_assets:
    d = scan_adx(t, "4h", "2y", label)
    if d:
        s = score_asset(d)
        res4.append((s, d))

res4.sort(key=lambda x: x[0], reverse=True)

for i, (s, d) in enumerate(res4):
    e = "🔥" if s >= 70 else ("✅" if s >= 55 else ("⚠️" if s >= 40 else "❌"))
    print(f"  {e} {i+1:<2} {d['label']:<12} {d['avg_adx']:<5.1f} {d['adx_gt20']:<4.0f}% {d['entry_freq']:<5.2f}% {d['pdi_bias']:<4.0f}% {d['bars']:<5} {s:<3.0f}")

# ── Best per asset ──
print(f"\n\n{'='*65}")
print("  🏆 FINAL RANKING (best TF per asset)")
print(f"{'='*65}")
print(f"  {'Rank':<5} {'Asset':<12} {'TF':<5} {'Score':<6} {'ADX':<6} {'Entry%':<7}")
print(f"  {'─'*4} {'─'*11} {'─'*4} {'─'*5} {'─'*5} {'─'*6}")

best_map = {}
for t, label in all_assets:
    best = None
    for rl, tf_name in [(res1, "1H"), (res4, "4H")]:
        for s, d in rl:
            if d["label"] == label:
                if best is None or s > best[0]:
                    best = (s, tf_name, d)
    if best:
        best_map[label] = best

ranked = sorted(best_map.items(), key=lambda x: x[1][0], reverse=True)
for i, (label, (s, tf, d)) in enumerate(ranked):
    e = "🏆" if i < 5 else ("✅" if s >= 55 else ("⚠️" if s >= 40 else "❌"))
    print(f"  {e} {i+1:<3} {label:<12} {tf:<5} {s:<4.0f}/90 {d['avg_adx']:<5.1f} {d['entry_freq']:.1f}%")

print("\n✅ DONE")
