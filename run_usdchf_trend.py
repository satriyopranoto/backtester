"""
USD/CHF 1h — Basis+ADX Single TF + CFD (0.05 lot)
"""
import numpy as np, pandas as pd, yfinance as yf, warnings
warnings.filterwarnings('ignore')

print("Loading USD/CHF 1h...")
t = 'USDCHF=X'
h1 = yf.download(t, period='1y', interval='1h', progress=False)
if isinstance(h1.columns, pd.MultiIndex): h1.columns = h1.columns.get_level_values(0)
c = h1['Close'].values.astype(float); h = h1['High'].values.astype(float); l = h1['Low'].values.astype(float)
print(f"  {len(h1)} bars")

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
atr10 = pd.Series(tr).rolling(10).mean().iloc[-1]

# CFD Config
MODAL = 1000.0; LEVERAGE = 100; LOT = 0.05; STOP_OUT = 30
CONTRACT = 100000 * LOT

print(f"Modal: $1,000 | Lot: {LOT} | Leverage: 1:{LEVERAGE}")
print(f"Harga: {c[-1]:.5f} | ATR: {atr10*10000:.0f} pips")
print(f"Nilai per pip: ${CONTRACT*0.0001:.2f}")

def backtest():
    balance = MODAL; pos = None; eq = []; trades = []; mc = 0; n = len(c)
    for i in range(30, n):
        if i % 1000 == 0: print(f"    {i}/{n}...")
        p = c[i]
        
        if pos:
            um = (pos['ep'] * CONTRACT * pos['sz']) / LEVERAGE
            fp = (p - pos['ep']) * CONTRACT * pos['sz']
            eq_ = balance + fp; ml = (eq_ / um) * 100
            
            if ml < STOP_OUT:
                bal_ret = (pos['ep'] * CONTRACT * pos['sz']) / LEVERAGE
                balance += fp + bal_ret; balance = max(balance, 0)
                trades.append({'r': (p/pos['ep']-1)*100, 'ex': 'MC'}); mc+=1; pos=None
                eq.append(balance); continue
            
            if p < pos['sl']:
                bal_ret = (pos['ep'] * CONTRACT * pos['sz']) / LEVERAGE
                balance += fp + bal_ret; balance = max(balance, 0)
                trades.append({'r': (p/pos['ep']-1)*100, 'ex': 'SL'}); pos=None
                eq.append(balance); continue
            
            fl = (p/pos['ep']-1)*100
            if fl > pos['tp']:
                bal_ret = (pos['ep'] * CONTRACT * pos['sz']) / LEVERAGE
                balance += fp + bal_ret; balance = max(balance, 0)
                trades.append({'r': (p/pos['ep']-1)*100, 'ex': 'TP'}); pos=None
                eq.append(balance); continue
        
        if pos is None and balance > 0:
            buy = (h[i] > sl[i] and c[i] > sma20[i] and adx[i] > 20 and pdi[i] > mdi[i]
                   and pdi[i] > pdi[max(0,i-5)] and not np.isnan(sl[i]) and sl[i] > 0)
            if buy:
                um = (p * CONTRACT * LOT) / LEVERAGE
                if um <= balance:
                    sd = abs(p - sl[i])
                    pos = {'ep': p, 'sl': sl[i], 'tp': 0.4*sd/p*100, 'sz': LOT}
                    balance -= um
        
        ev = balance + (0 if pos is None else (p - pos['ep']) * CONTRACT * pos['sz'])
        eq.append(ev)
    
    if pos:
        bal_ret = (pos['ep'] * CONTRACT * pos['sz']) / LEVERAGE
        fp = (c[-1] - pos['ep']) * CONTRACT * pos['sz']
        balance += fp + bal_ret; balance = max(balance, 0)
    
    fe = eq[-1]; yrs = len(eq)/(252*7)
    r = {'ret': (fe-MODAL)/MODAL*100, 'n': len(trades), 'final': fe, 'mc': mc}
    if trades:
        dft = pd.DataFrame(trades); w = dft[dft['r']>0]; l_ = dft[dft['r']<=0]
        r['wr'] = len(w)/len(dft)*100; r['best'] = dft['r'].max(); r['worst'] = dft['r'].min()
        r['pf'] = abs(w['r'].sum()/l_['r'].sum()) if len(l_)>0 and l_['r'].sum()!=0 else float('inf')
        eqa = np.array(eq); pk = np.maximum.accumulate(eqa)
        r['dd'] = ((pk-eqa)/pk*100).max()
        dr = pd.Series(eq).pct_change().dropna()
        r['sharpe'] = np.sqrt(252*7)*dr.mean()/dr.std() if dr.std()>0 else 0
        r['cagr'] = ((fe/MODAL)**(1/yrs)-1)*100 if yrs>0 else 0
        dd_all = ((np.maximum.accumulate(eqa)-eqa)/np.maximum.accumulate(eqa))*100
        r['ulcer'] = np.sqrt(np.mean(dd_all**2))
        r['martin'] = r['cagr']/r['ulcer'] if r['ulcer'] > 0 else 0
    return r

r = backtest()

print(f"\n{'='*40}")
print(f"  USD/CHF — Basis+ADX Single TF")
print(f"{'='*40}")
print(f"  Return    : {r['ret']:+.2f}%")
print(f"  CAGR      : {r.get('cagr',0):+.2f}%/yr")
print(f"  Sharpe    : {r.get('sharpe',0):.2f}")
print(f"  Ulcer     : {r.get('ulcer',0):.2f}")
print(f"  Max DD    : -{r.get('dd',0):.2f}%")
print(f"  Trades    : {r['n']} | WR: {r.get('wr',0):.1f}%")
print(f"  PF        : {r.get('pf',0):.2f}")
print(f"  MC        : {r.get('mc',0)}x")
print(f"  Final     : ${r['final']:.2f}")
print(f"  Best/Worst: +{r.get('best',0):.2f}% / {r.get('worst',0):.2f}%")
