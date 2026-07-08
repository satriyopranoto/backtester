"""
IBM 30m — Fast Backtest (numpy-based, 1998-2026)
Single TF vs Multi TF (30m + Daily PDI>MDI)
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

print("Loading IBM 30m...")
df = pd.read_csv(r'C:\Users\Acer\AppData\Local\hermes\cache\documents\doc_dbdfa09eafdf_IBM_30min.txt',
    parse_dates=['DateTime'], dayfirst=True)
df = df.rename(columns={'DateTime':'datetime','Open':'Open','High':'High','Low':'Low','Close':'Close','Volume':'Volume'})
df = df.set_index('datetime').sort_index()
df.index = pd.to_datetime(df.index)
print(f"  {len(df)} bars | {df.index[0]} → {df.index[-1]}")

def calc_i(c, h, l):
    tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
    up, dn = np.diff(h, prepend=h[0]), np.diff(l, prepend=l[0])
    pdm = np.where((up>dn)&(up>0), up, 0); mdm = np.where((dn>up)&(dn>0), dn, 0)
    atr = pd.Series(tr).rolling(14).mean().values
    sp = pd.Series(pdm).rolling(14).mean().values; sm = pd.Series(mdm).rolling(14).mean().values
    pdi = np.where(atr>0, 100*sp/atr, 0); mdi = np.where(atr>0, 100*sm/atr, 0)
    dx = np.where((pdi+mdi)>0, 100*np.abs(pdi-mdi)/(pdi+mdi), 0)
    adx = pd.Series(dx).rolling(14).mean().values
    return adx, pdi, mdi

c = df['Close'].values.astype(float); h = df['High'].values.astype(float); l = df['Low'].values.astype(float)
sma20 = pd.Series(c).rolling(20).mean().values
tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
sl = pd.Series(c).rolling(28).max().values - 2*pd.Series(tr).rolling(10).mean().values*(2.8-1)/2.8
adx, pdi, mdi = calc_i(c, h, l)

# Daily HTF from resampled
daily = df.resample('D').agg({'Close':'last','High':'max','Low':'min'}).dropna()
dc, dh, dl = daily['Close'].values.astype(float), daily['High'].values.astype(float), daily['Low'].values.astype(float)
ddates = daily.index
_, dp, dm = calc_i(dc, dh, dl)

# Map daily to 30m using merge_asof (fast)
d_map = pd.DataFrame({'date': ddates, 'd_pdi': dp, 'd_mdi': dm}).set_index('date')
df_30m = pd.DataFrame({'date': df.index.normalize()}).sort_values('date')
df_30m = pd.merge_asof(df_30m, d_map, left_on='date', right_index=True, direction='backward')
df['D_PDI'], df['D_MDI'] = df_30m['d_pdi'].values, df_30m['d_mdi'].values
d_pdi, d_mdi = df['D_PDI'].values, df['D_MDI'].values

# Backtest function
def backtest(multi_tf=False):
    name = "Multi TF" if multi_tf else "Single TF"
    print(f"\n  {name}...")
    cash = 100000.0; pos = None; eq = []; trades = []
    
    for i in range(30, len(c)):
        if i % 20000 == 0: print(f"    {i}/{len(c)}...")
        
        # Exit
        if pos is not None:
            if c[i] < pos['sl_price']:
                cash += pos['size'] * c[i]
                r = (c[i]/pos['entry_price']-1)*100
                trades.append(r)
                pos = None
            else:
                fl = (c[i]/pos['entry_price']-1)*100
                if fl > pos['tp_pct']:
                    cash += pos['size'] * c[i]
                    r = (c[i]/pos['entry_price']-1)*100
                    trades.append(r)
                    pos = None
        
        # Entry
        if pos is None:
            if (h[i] > sl[i] and c[i] > sma20[i] and adx[i] > 20 and pdi[i] > mdi[i]
                and pdi[i] > pdi[i-min(6,i)]):
                htf_ok = d_pdi[i] > d_mdi[i] if multi_tf else True
                if htf_ok and not np.isnan(sl[i]) and sl[i] > 0:
                    sd = abs(c[i] - sl[i])
                    if sd > 0:
                        sz = max(1, int((cash*0.01)/sd))
                        cost = sz * c[i]
                        if sz > 0 and cost <= cash * 0.95:
                            pos = {'size': sz, 'entry_price': c[i], 'sl_price': sl[i], 'tp_pct': 0.4*sd/c[i]*100}
                            cash -= cost
        
        eq.append(cash + (pos['size']*c[i] if pos else 0))
    
    if pos:
        cash += pos['size'] * c[-1]
    
    fe = eq[-1]; yrs = len(eq)/252/7
    r = {'ret': (fe-100000)/100000*100, 'n': len(trades), 'final': fe}
    if trades:
        ta = np.array(trades)
        wins = ta[ta>0]; losses = ta[ta<=0]
        r['wr'] = len(wins)/len(ta)*100
        r['best'] = ta.max(); r['worst'] = ta.min()
        r['pf'] = abs(wins.sum()/losses.sum()) if len(losses)>0 and losses.sum()!=0 else float('inf')
        eqa = np.array(eq); pk = np.maximum.accumulate(eqa)
        r['dd'] = ((pk-eqa)/pk*100).max()
        dr = pd.Series(eq).pct_change().dropna()
        r['sharpe'] = np.sqrt(252*7)*dr.mean()/dr.std() if dr.std()>0 else 0
        r['cagr'] = ((fe/100000)**(1/yrs)-1)*100 if yrs>0 else 0
    return r

for mtf in [False, True]:
    r = backtest(mtf)
    lbl = "Multi TF (30m+Daily)" if mtf else "Single TF (30m)"
    print(f"\n{'='*40}")
    print(f"  {lbl}")
    print(f"{'='*40}")
    print(f"  Return   : {r['ret']:+.2f}%")
    print(f"  CAGR     : {r.get('cagr',0):+.2f}%/yr")
    print(f"  Sharpe   : {r.get('sharpe',0):.2f}")
    print(f"  Max DD   : -{r.get('dd',0):.2f}%")
    print(f"  Trades   : {r['n']} | WR: {r.get('wr',0):.1f}%")
    print(f"  Best/Worst: +{r.get('best',0):.2f}% / {r.get('worst',0):.2f}%")
