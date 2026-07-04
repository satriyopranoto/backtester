"""
Multi-Timeframe Basis+ADX Strategy
Buy logic = existing intraday conditions + higher TF confirmation:
  close > basis_d  AND  pdi_d > mdi_d

suffix _d = one TF level up:
  intraday (30m/1h) → daily
  daily → weekly
"""
import numpy as np, pandas as pd, yfinance as yf, warnings
warnings.filterwarnings("ignore")

def calc_adx_full(c, h, l, p=14):
    tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
    up, dn = np.diff(h, prepend=h[0]), np.diff(l, prepend=l[0])
    pdm = np.where((up>dn)&(up>0), up, 0)
    mdm = np.where((dn>up)&(dn>0), dn, 0)
    atr_s = pd.Series(tr).rolling(p).mean().values
    sp = pd.Series(pdm).rolling(p).mean().values
    sm = pd.Series(mdm).rolling(p).mean().values
    pdi = np.where(atr_s>0, 100*sp/atr_s, 0)
    mdi = np.where(atr_s>0, 100*sm/atr_s, 0)
    dx = np.where((pdi+mdi)>0, 100*np.abs(pdi-mdi)/(pdi+mdi), 0)
    adx = pd.Series(dx).rolling(p).mean().values
    return adx, pdi, mdi, sp, sm

def get_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty or len(df) < 30: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    c = df['Close'].values.astype(float)
    h = df['High'].values.astype(float)
    l = df['Low'].values.astype(float)
    v = df['Volume'].values.astype(float) if 'Volume' in df.columns else np.zeros(len(df))
    
    sma20 = pd.Series(c).rolling(20).mean().values
    adx, pdi, mdi, _, _ = calc_adx_full(c, h, l)
    
    return {'df': df, 'close': c, 'high': h, 'low': l, 'volume': v,
            'sma20': sma20, 'adx': adx, 'pdi': pdi, 'mdi': mdi}

# ── Download both TFs ──
print("=== MULTI-TIMEFRAME ANALYSIS ===")
print("Current TF: 30 menit, Higher TF: Daily\n")

# Current TF (30m) — need ~1 month for enough bars
cur = get_data('GBPUSD=X', '1mo', '30m')
# Higher TF (daily) — need enough daily data to align
htf = get_data('GBPUSD=X', '6mo', '1d')

if cur is None or htf is None:
    print("❌ Gagal download data")
    exit()

n = len(cur['close'])
print(f"Current TF (30m): {n} bars | {cur['df'].index[0]} → {cur['df'].index[-1]}")
print(f"Higher TF (daily): {len(htf['close'])} bars | {htf['df'].index[0]} → {htf['df'].index[-1]}")

# ── Map daily data to each 30m bar ──
# For each 30m bar, find the corresponding daily bar (same date)
cur_dates = pd.Series(cur['df'].index.normalize())
htf_dates = pd.Series(htf['df'].index.normalize())

# Build lookup: date -> daily indicators
htf_lookup = {}
for i, d in enumerate(htf_dates):
    htf_lookup[d] = {
        'sma20': htf['sma20'][i],
        'pdi': htf['pdi'][i],
        'mdi': htf['mdi'][i],
        'close': htf['close'][i],
        'adx': htf['adx'][i],
    }

# For each 30m bar, check the daily condition
print("\n=== Multi-TF Check Sample (last 20 bars) ===")
print(f"{'Date':20s} {'30m Close':10s} {'ADX':5s} {'+DI':5s} {'-DI':5s} {'30mOK?':7s} {'Dly SMA':8s} {'Dly+DI':5s} {'Dly-DI':5s} {'DlyOK?':7s} {'BUY?':5s}")
print("-"*95)

signals = 0
multi_tf_signals = 0
for i in range(max(20, n-20), n):
    d = cur['df'].index[i].normalize()
    close_cur = cur['close'][i]
    adx_cur = cur['adx'][i]
    pdi_cur = cur['pdi'][i]
    mdi_cur = cur['mdi'][i]
    sma_cur = cur['sma20'][i]
    
    # Current TF buy signal (simplified)
    cur_ok = (pdi_cur > mdi_cur and adx_cur > 20 and close_cur > sma_cur 
              and not np.isnan(adx_cur))
    
    # Higher TF confirmation
    htf_data = htf_lookup.get(d)
    htf_ok = False
    if htf_data and not np.isnan(htf_data['pdi']):
        htf_ok = (close_cur > htf_data['sma20'] and htf_data['pdi'] > htf_data['mdi'])
    
    buy = cur_ok and htf_ok
    if cur_ok: signals += 1
    if buy: multi_tf_signals += 1
    
    if i >= n-10 or buy:  # Show last 10 bars and all buy signals
        cur_str = "✅" if cur_ok else "❌"
        htf_sma = f"{htf_data['sma20']:.5f}" if htf_data else "N/A"
        htf_pdi = f"{htf_data['pdi']:.1f}" if htf_data else "N/A"
        htf_mdi = f"{htf_data['mdi']:.1f}" if htf_data else "N/A"
        htf_str = "✅" if htf_ok else "❌"
        buy_str = "🚀" if buy else ""
        d_str = d.strftime('%Y-%m-%d')
        print(f"{d_str:20s} {close_cur:10.5f} {adx_cur:5.1f} {pdi_cur:5.1f} {mdi_cur:5.1f} {cur_str:7s} {htf_sma:8s} {htf_pdi:5s} {htf_mdi:5s} {htf_str:7s} {buy_str:5s}")

print(f"\n=== SUMMARY ===")
print(f"  Current TF signals (30m):       {signals}")
print(f"  Multi-TF signals (30m + daily): {multi_tf_signals}")
print(f"  Filter reduction:               {100 - (multi_tf_signals/signals*100 if signals else 0):.0f}% fewer signals")
print()
print("Multi-TF hanya masuk saat DAILY juga bullish (close>sma20 & +DI>-DI)")
print("Ini menyaring sinyal palsu di TF lebih rendah.")
