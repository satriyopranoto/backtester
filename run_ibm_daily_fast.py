"""
IBM Daily — Fast Backtest, 1998-2026
Single TF Basis+ADX
Compare with 30m results
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

print("Loading IBM Daily...")
df = pd.read_csv(r'C:\Users\Acer\AppData\Local\hermes\cache\documents\doc_d213ef5ad698_IBM_1d.txt',
    parse_dates=['DateTime'], dayfirst=True)
df = df.rename(columns={'DateTime':'date','Open':'Open','High':'High','Low':'Low','Close':'Close','Volume':'Volume'})
df = df.set_index('date').sort_index()
df.index = pd.to_datetime(df.index)
print(f"  {len(df)} bars | {df.index[0].date()} → {df.index[-1].date()}")

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

# Backtest
print("\n  Running Single TF (Daily) backtest...")
cash = 100000.0; pos = None; eq = []; trades = []
n = len(c)

for i in range(30, n):
    if i % 1500 == 0: print(f"    {i}/{n}...")
    
    # Exit
    if pos is not None:
        if c[i] < pos['sl_price']:
            cash += pos['size'] * c[i]
            trades.append((c[i]/pos['entry_price']-1)*100)
            pos = None
        else:
            fl = (c[i]/pos['entry_price']-1)*100
            if fl > pos['tp_pct']:
                cash += pos['size'] * c[i]
                trades.append((c[i]/pos['entry_price']-1)*100)
                pos = None
    
    # Entry
    if pos is None:
        if (h[i] > sl[i] and c[i] > sma20[i] and adx[i] > 20 and pdi[i] > mdi[i]
            and pdi[i] > pdi[i-min(6,i)] and not np.isnan(sl[i]) and sl[i] > 0):
            sd = abs(c[i] - sl[i])
            if sd > 0:
                sz = max(1, int((cash*0.01)/sd))
                cost = sz * c[i]
                if sz > 0 and cost <= cash * 0.95:
                    pos = {'size': sz, 'entry_price': c[i], 'sl_price': sl[i], 'tp_pct': 0.4*sd/c[i]*100}
                    cash -= cost
    
    eq.append(cash + (pos['size']*c[i] if pos else 0))

if pos: cash += pos['size'] * c[-1]

fe = eq[-1]; yrs = len(eq)/252
r = {'ret':(fe-100000)/100000*100, 'n':len(trades), 'final':fe}
if trades:
    ta = np.array(trades)
    wins = ta[ta>0]; losses = ta[ta<=0]
    r['wr'] = len(wins)/len(ta)*100
    r['best'] = ta.max(); r['worst'] = ta.min()
    r['pf'] = abs(wins.sum()/losses.sum()) if len(losses)>0 and losses.sum()!=0 else float('inf')
    eqa = np.array(eq); pk = np.maximum.accumulate(eqa)
    r['dd'] = ((pk-eqa)/pk*100).max()
    dr = pd.Series(eq).pct_change().dropna()
    r['sharpe'] = np.sqrt(252)*dr.mean()/dr.std() if dr.std()>0 else 0
    r['cagr'] = ((fe/100000)**(1/yrs)-1)*100 if yrs>0 else 0

print(f"\n{'='*40}")
print(f"  IBM Daily — Basis+ADX (1998-2026)")
print(f"{'='*40}")
print(f"  Return   : {r['ret']:+.2f}%")
print(f"  CAGR     : {r.get('cagr',0):+.2f}%/yr")
print(f"  Sharpe   : {r.get('sharpe',0):.2f}")
print(f"  Max DD   : -{r.get('dd',0):.2f}%")
print(f"  Trades   : {r['n']} | WR: {r.get('wr',0):.1f}%")
print(f"  Best/Worst: +{r.get('best',0):.2f}% / {r.get('worst',0):.2f}%")

print(f"\n{'='*40}")
print(f"  PERBANDINGAN IBM")
print(f"{'='*40}")
print(f"  {'Timeframe':15s} {'Trades':8s} {'WR':8s} {'Sharpe':8s} {'DD':8s}")
print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
print(f"  {'30m Single TF':15s} {2656:8d} {66.6:7.1f}% {1.23:7.2f} {-97.76:7.2f}%")
print(f"  {'30m Multi TF':15s} {1425:8d} {66.9:7.1f}% {1.04:7.2f} {-91.12:7.2f}%")
print(f"  {'Daily Single':15s} {r['n']:8d} {r.get('wr',0):7.1f}% {r.get('sharpe',0):7.2f} {-r.get('dd',0):7.2f}%")
