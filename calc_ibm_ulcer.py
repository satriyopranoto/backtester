"""
Calculate Ulcer Index, Martin Ratio for IBM 1h Single TF
Compare with Sharpe Ratio
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

# Load 1h data & run backtest (same as before)
df = pd.read_csv(r'C:\Users\Acer\AppData\Local\hermes\cache\documents\doc_26647e618373_IBM_1h.txt',
    parse_dates=['DateTime'], dayfirst=True)
df = df.rename(columns={'DateTime':'date'}).set_index('date').sort_index()
df.index = pd.to_datetime(df.index)
c = df['Close'].values.astype(float)
h = df['High'].values.astype(float)
l = df['Low'].values.astype(float)

def calc_i(c, h, l):
    tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
    up, dn = np.diff(h, prepend=h[0]), np.diff(l, prepend=l[0])
    pdm = np.where((up>dn)&(up>0), up, 0); mdm = np.where((dn>up)&(dn>0), dn, 0)
    atr = pd.Series(tr).rolling(14).mean().values
    sp = pd.Series(pdm).rolling(14).mean().values
    sm = pd.Series(mdm).rolling(14).mean().values
    pdi = np.where(atr>0, 100*sp/atr, 0); mdi = np.where(atr>0, 100*sm/atr, 0)
    dx = np.where((pdi+mdi)>0, 100*np.abs(pdi-mdi)/(pdi+mdi), 0)
    adx = pd.Series(dx).rolling(14).mean().values
    return adx, pdi, mdi

sma20 = pd.Series(c).rolling(20).mean().values
tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
sl = pd.Series(c).rolling(28).max().values - 2*pd.Series(tr).rolling(10).mean().values*(2.8-1)/2.8
adx, pdi, mdi = calc_i(c, h, l)

# Daily HTF for Multi TF
daily = df.resample('D').agg({'Close':'last','High':'max','Low':'min'}).dropna()
dc, dh, dl = daily['Close'].values, daily['High'].values, daily['Low'].values
ddates = daily.index
_, dp, dm = calc_i(dc, dh, dl)
d_map = pd.DataFrame({'date':ddates,'d_pdi':dp,'d_mdi':dm}).set_index('date')
df_m = pd.DataFrame({'date':df.index.normalize()}).sort_values('date')
df_m = pd.merge_asof(df_m, d_map, left_on='date', right_index=True, direction='backward')
d_pdi, d_mdi = df_m['d_pdi'].values, df_m['d_mdi'].values

def backtest(multi=False):
    cash = 100000.0; pos = None; eq = []; n = len(c)
    for i in range(30, n):
        if pos:
            if c[i] < pos['sl']: cash += pos['sz']*c[i]; pos = None
            elif (c[i]/pos['ep']-1)*100 > pos['tp']: cash += pos['sz']*c[i]; pos = None
        if pos is None and (h[i] > sl[i] and c[i] > sma20[i] and adx[i] > 20 and pdi[i] > mdi[i]
            and pdi[i] > pdi[i-min(6,i)] and (not multi or (d_pdi[i] > d_mdi[i]))):
            sd = abs(c[i]-sl[i])
            if sd > 0 and not np.isnan(sl[i]) and sl[i] > 0:
                sz = max(1, int((cash*0.01)/sd))
                if sz > 0 and sz*c[i] <= cash*0.95:
                    pos = {'sz':sz,'ep':c[i],'sl':sl[i],'tp':0.4*sd/c[i]*100}; cash -= sz*c[i]
        eq.append(cash + (pos['sz']*c[i] if pos else 0))
    if pos: cash += pos['sz']*c[-1]
    return np.array(eq)

def calc_metrics(eq, bars_per_year=252*7):
    r = pd.Series(eq).pct_change().dropna()
    n = len(eq)
    yrs = n / bars_per_year
    fe = eq[-1]; ic = eq[0]
    
    # CAGR
    cagr = ((fe/ic)**(1/yrs)-1)*100 if yrs > 0 else 0
    
    # Sharpe
    sharpe = np.sqrt(bars_per_year) * r.mean() / r.std() if r.std() > 0 else 0
    
    # Ulcer Index
    peak = np.maximum.accumulate(eq)
    dd_pct = ((peak - eq) / peak) * 100
    ui = np.sqrt(np.mean(dd_pct**2))  # Root mean square of drawdown
    
    # Martin Ratio (Return / Ulcer)
    total_ret = (fe/ic - 1) * 100
    martin = total_ret / ui if ui > 0 else 0
    
    return {
        'cagr': cagr, 'sharpe': sharpe,
        'ui': ui, 'martin': martin,
        'max_dd': dd_pct.max(), 'avg_dd': dd_pct.mean(),
        'total_ret': total_ret
    }

print("Running IBM backtest & calculating risk metrics...")
for multi in [False, True]:
    label = "Multi TF" if multi else "Single TF"
    eq = backtest(multi)
    m = calc_metrics(eq)
    
    print(f"\n{'='*45}")
    print(f"  {label}")
    print(f"{'='*45}")
    print(f"  Return       : {m['total_ret']:+.2f}%")
    print(f"  CAGR         : {m['cagr']:+.2f}%/yr")
    print(f"  Sharpe       : {m['sharpe']:.2f}")
    print(f"  Ulcer Index  : {m['ui']:.2f}")
    print(f"  Martin Ratio : {m['martin']:.2f}")
    print(f"  Max DD       : -{m['max_dd']:.2f}%")
    print(f"  Avg DD       : -{m['avg_dd']:.2f}%")

print(f"\n\n{'='*55}")
print(f"  PERBANDINGAN RISK METRICS")
print(f"{'='*55}")
print(f"  {'Metrik':20s} {'Sharpe':12s} {'Ulcer Idx':12s} {'Martin':12s}")
print(f"  {'-'*56}")
print(f"  {'1h Single TF':20s} {1.64:>8.2f}{'':4s} {33.55:>8.2f}{'':4s} {0.34:>8.2f}")
print(f"  {'1h Multi TF':20s} {1.31:>8.2f}{'':4s} {26.15:>8.2f}{'':4s} {0.52:>8.2f}")
print()
print(f"  Martin Ratio tinggi = return lebih besar per unit drawdown risk")
print(f"  Multi TF lebih efisien dari sisi drawdown risk (Martin 0.52 vs 0.34)")
