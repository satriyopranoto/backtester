"""
IBM 1h — Fast Backtest (1998-2026)
Compare: Single TF vs Multi TF, vs 30m results
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

print("Loading IBM 1h...")
df = pd.read_csv(r'C:\Users\Acer\AppData\Local\hermes\cache\documents\doc_26647e618373_IBM_1h.txt',
    parse_dates=['DateTime'], dayfirst=True)
df = df.rename(columns={'DateTime':'date','Open':'Open','High':'High','Low':'Low','Close':'Close','Volume':'Volume'})
df = df.set_index('date').sort_index()
df.index = pd.to_datetime(df.index)
print(f"  {len(df)} bars | {df.index[0]} → {df.index[-1]}")

c = df['Close'].values.astype(float); h = df['High'].values.astype(float); l = df['Low'].values.astype(float)

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

sma20 = pd.Series(c).rolling(20).mean().values
tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
sl = pd.Series(c).rolling(28).max().values - 2*pd.Series(tr).rolling(10).mean().values*(2.8-1)/2.8
adx, pdi, mdi = calc_i(c, h, l)

# Daily HTF
daily = df.resample('D').agg({'Close':'last','High':'max','Low':'min'}).dropna()
dc, dh, dl = daily['Close'].values, daily['High'].values, daily['Low'].values
ddates = daily.index
_, dp, dm = calc_i(dc, dh, dl)

d_map = pd.DataFrame({'date': ddates, 'd_pdi': dp, 'd_mdi': dm}).set_index('date')
df_30m = pd.DataFrame({'date': df.index.normalize()}).sort_values('date')
df_30m = pd.merge_asof(df_30m, d_map, left_on='date', right_index=True, direction='backward')
d_pdi, d_mdi = df_30m['d_pdi'].values, df_30m['d_mdi'].values

def backtest(multi=False):
    lbl = "Multi TF" if multi else "Single TF"
    print(f"\n  Running {lbl}...")
    cash = 100000.0; pos = None; eq = []; trades = []; n = len(c)
    for i in range(30, n):
        if i % 10000 == 0: print(f"    {i}/{n}...")
        if pos:
            if c[i] < pos['sl']:
                cash += pos['sz']*c[i]; trades.append((c[i]/pos['ep']-1)*100); pos = None
            elif (c[i]/pos['ep']-1)*100 > pos['tp']:
                cash += pos['sz']*c[i]; trades.append((c[i]/pos['ep']-1)*100); pos = None
        if pos is None and (h[i] > sl[i] and c[i] > sma20[i] and adx[i] > 20 and pdi[i] > mdi[i]
            and pdi[i] > pdi[i-min(6,i)] and (not multi or (d_pdi[i] > d_mdi[i]))):
            sd = abs(c[i]-sl[i])
            if sd > 0 and not np.isnan(sl[i]) and sl[i] > 0:
                sz = max(1, int((cash*0.01)/sd))
                if sz > 0 and sz*c[i] <= cash*0.95:
                    pos = {'sz':sz,'ep':c[i],'sl':sl[i],'tp':0.4*sd/c[i]*100}; cash -= sz*c[i]
        eq.append(cash + (pos['sz']*c[i] if pos else 0))
    if pos: cash += pos['sz']*c[-1]
    fe = eq[-1]; yrs = len(eq)/(252*7)
    r = {'ret':(fe-100000)/100000*100,'n':len(trades)}
    if trades:
        ta = np.array(trades); w=ta[ta>0]; l=ta[ta<=0]
        r['wr']=len(w)/len(ta)*100; r['best']=ta.max(); r['worst']=ta.min()
        r['pf']=abs(w.sum()/l.sum()) if len(l)>0 and l.sum()!=0 else float('inf')
        pk=np.maximum.accumulate(np.array(eq)); r['dd']=((pk-np.array(eq))/pk*100).max()
        dr=pd.Series(eq).pct_change().dropna()
        r['sharpe']=np.sqrt(252*7)*dr.mean()/dr.std() if dr.std()>0 else 0
        r['cagr']=((fe/100000)**(1/yrs)-1)*100 if yrs>0 else 0
    print(f"  Return: {r['ret']:+.2f}% | Trades: {r['n']} | WR: {r.get('wr',0):.1f}% | Sharpe: {r.get('sharpe',0):.2f} | DD: -{r.get('dd',0):.2f}%")
    return r

results = []
for mtf in [False, True]:
    lbl = "Multi TF (1h+Daily)" if mtf else "Single TF (1h)"
    r = backtest(mtf)
    results.append((lbl, r))
    print(f"\n{'='*40}")
    print(f"  {lbl}")
    print(f"{'='*40}")
    print(f"  Return: {r['ret']:+.2f}% | CAGR: {r.get('cagr',0):+.2f}%/yr")
    print(f"  Sharpe: {r.get('sharpe',0):.2f} | DD: -{r.get('dd',0):.2f}%")
    print(f"  Trades: {r['n']} | WR: {r.get('wr',0):.1f}% | PF: {r.get('pf',0):.2f}")
    print(f"  Best/Worst: +{r.get('best',0):.2f}% / {r.get('worst',0):.2f}%")

print(f"\n\n{'='*50}")
print(f"  PERBANDINGAN IBM — Semua Timeframe")
print(f"{'='*50}")
print(f"  {'Timeframe':20s} {'Trades':8s} {'WR':8s} {'Sharpe':7s} {'DD':8s}")
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*7} {'-'*8}")
# 1h results
for lbl, r in results:
    print(f"  {lbl:20s} {r['n']:8d} {r.get('wr',0):7.1f}% {r.get('sharpe',0):7.2f} {-r.get('dd',0):7.2f}%")
# 30m results from previous run
print(f"  {'30m Single TF':20s} {2656:8d} {66.6:7.1f}% {1.23:7.2f} {-97.76:7.2f}%")
print(f"  {'30m Multi TF':20s} {1425:8d} {66.9:7.1f}% {1.04:7.2f} {-91.12:7.2f}%")
print(f"  {'Daily Single':20s} {89:8d} {14.6:7.1f}% {-1.06:7.2f} {-99.92:7.2f}%")
