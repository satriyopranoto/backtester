"""Crypto Long Opportunity Scanner — ADX(14) + SMA20 Framework.
Mencari crypto pair dengan setup long terbaik saat ini."""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

CRYPTO = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD",
    "DOGE-USD", "AVAX-USD", "DOT-USD", "LINK-USD", "MATIC-USD",
    "ATOM-USD", "UNI7083-USD", "APT-USD", "ARB-USD", "OP-USD",
    "INJ-USD", "TIA-USD", "SEI-USD", "SUI-USD", "PEPE-USD",
    "FET-USD", "RUNE-USD", "AAVE-USD", "MKR-USD", "CRV-USD",
]

results = []

for ticker in CRYPTO:
    try:
        df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        # SMA20
        df['sma20'] = df['close'].rolling(20).mean()
        
        # ADX(14)
        high, low, close = df['high'], df['low'], df['close']
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()
        period = 14
        atr = tr.rolling(period).mean()
        smoothed_plus_dm = plus_dm.rolling(period).mean()
        smoothed_minus_dm = minus_dm.rolling(period).mean()
        pdi = 100 * smoothed_plus_dm / atr
        mdi = 100 * smoothed_minus_dm / atr
        dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
        df['adx'] = dx.rolling(period).mean()
        df['pdi'] = pdi
        df['mdi'] = mdi
        
        valid = df.dropna()
        if len(valid) < 50:
            continue
        
        # ── Full period stats ──
        total = 0
        bull = 0  # ADX>25 + Close>SMA20
        bear = 0  # ADX>25 + Close<SMA20
        side = 0  # ADX<=25
        for i in range(20, len(valid)):
            if np.isnan(valid['adx'].iloc[i]) or np.isnan(valid['sma20'].iloc[i]):
                continue
            total += 1
            c = float(valid['close'].iloc[i])
            s = float(valid['sma20'].iloc[i])
            a = float(valid['adx'].iloc[i])
            if a > 25:
                if c > s:
                    bull += 1
                else:
                    bear += 1
            else:
                side += 1
        
        if total == 0:
            continue
        
        bull_pct = bull / total * 100
        bear_pct = bear / total * 100
        side_pct = side / total * 100
        
        if bull_pct > 35:
            cls = "STRONG BULLISH"
        elif bull_pct >= 30:
            cls = "WEAK BULLISH"
        elif bear_pct > 35:
            cls = "STRONG BEARISH"
        elif bear_pct >= 30:
            cls = "WEAK BEARISH"
        else:
            cls = "SIDEWAYS"
        
        # ── Last 100 bars ──
        l100_start = max(20, len(valid) - 100)
        l100_total = 0
        l100_bull = 0
        for i in range(l100_start, len(valid)):
            if np.isnan(valid['adx'].iloc[i]) or np.isnan(valid['sma20'].iloc[i]):
                continue
            l100_total += 1
            c = float(valid['close'].iloc[i])
            s = float(valid['sma20'].iloc[i])
            a = float(valid['adx'].iloc[i])
            if a > 25 and c > s:
                l100_bull += 1
        
        l100_bull_pct = l100_bull / l100_total * 100 if l100_total > 0 else 0
        
        if l100_bull_pct > 35:
            l100_cls = "STRONG BULLISH"
        elif l100_bull_pct >= 30:
            l100_cls = "WEAK BULLISH"
        else:
            l100_cls = "NOT BULLISH"
        
        # ── Current snapshot ──
        last = valid.iloc[-1]
        close_price = float(last['close'])
        sma20_val = float(last['sma20'])
        adx_val = float(last['adx'])
        pdi_val = float(last['pdi'])
        mdi_val = float(last['mdi'])
        sma20_dist = ((close_price / sma20_val) - 1) * 100
        
        # Long score (0-100)
        score = 0
        # Current above SMA20?
        if close_price > sma20_val:
            score += 20
            # How far above? (5-15% sweet spot)
            if 2 < sma20_dist < 20:
                score += 10
        # ADX strength
        if adx_val > 25:
            score += 20
        elif adx_val > 20:
            score += 10
        # +DI > -DI
        if pdi_val > mdi_val:
            score += 20
        # PDI rising (compare 5 bars ago)
        if len(valid) >= 6:
            pdi_5ago = float(valid['pdi'].iloc[-6])
            if pdi_val > pdi_5ago:
                score += 10
        # ADX rising
        if len(valid) >= 6:
            adx_5ago = float(valid['adx'].iloc[-6])
            if adx_val > adx_5ago:
                score += 10
        # Overall classification
        if "STRONG BULLISH" in cls:
            score += 10
        elif "WEAK BULLISH" in cls:
            score += 5
        
        results.append({
            'ticker': ticker,
            'price': round(close_price, 2),
            'sma20': round(sma20_val, 2),
            'sma20_dist': round(sma20_dist, 2),
            'adx': round(adx_val, 1),
            'pdi': round(pdi_val, 1),
            'mdi': round(mdi_val, 1),
            'class': cls,
            'bull_pct': round(bull_pct, 1),
            'bear_pct': round(bear_pct, 1),
            'side_pct': round(side_pct, 1),
            'l100': l100_cls,
            'l100_bull_pct': round(l100_bull_pct, 1),
            'score': score,
        })
        
        name = ticker.replace("-USD", "")
        print(f"  {name:>8} | ${close_price:>8,.2f} | {cls:<16} | ADX:{adx_val:>5.1f} | +DI:{pdi_val:>5.1f} -DI:{mdi_val:>5.1f} | SMA20:{sma20_dist:>+6.2f}% | L100:{l100_cls:<16} | Score:{score}")
        
    except Exception as e:
        continue

# Sort by score descending
results.sort(key=lambda x: x['score'], reverse=True)

print()
print("=" * 90)
print("  TOP LONG CANDIDATES — Ranked by Long Score")
print("=" * 90)
print(f"{'Rank':<5} {'Pair':<10} {'Price':>10} {'Class':<18} {'Score':<6} {'SMA20%':>8} {'ADX':>6} {'+DI/-DI':>12} {'L100'}")
print("-" * 90)
for i, r in enumerate(results[:10], 1):
    di = f"{r['pdi']}/{r['mdi']}"
    print(f"  #{i:<2}  {r['ticker']:<10} ${r['price']:>8,.2f} {r['class']:<18} {r['score']:<6} {r['sma20_dist']:>+7.2f}% {r['adx']:>5.1f} {di:>11} {r['l100']}")

print()
print("─" * 90)
print("LEGEND:")
print("  Score 0-100: price vs SMA20(30pts) + ADX strength(20pts) + direction(20pts) + momentum(30pts)")
print("  L100 = Last 100 bars classification")
print("  Best long setup: Price above SMA20, ADX>25, +DI>-DI, rising ADX/PDI")
