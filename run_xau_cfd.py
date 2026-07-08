"""
XAU/USD CFD — 1h, 1 tahun
Modal $1000, Leverage 1:100, Lot 0.1 fixed
MC Logic: Stop-Out di Margin Level 30%
Single TF vs Multi TF
"""
import numpy as np, pandas as pd, yfinance as yf, warnings
warnings.filterwarnings('ignore')

print("Loading XAU/USD...")
t = 'GC=F'
h1 = yf.download(t, period='1y', interval='1h', progress=False)
if isinstance(h1.columns, pd.MultiIndex): h1.columns = h1.columns.get_level_values(0)
print(f"  1h: {len(h1)} bars")
c = h1['Close'].values.astype(float); h = h1['High'].values.astype(float); l = h1['Low'].values.astype(float)

d = yf.download(t, period='1y', interval='1d', progress=False)
if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)

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
dc = d['Close'].values.astype(float); dh = d['High'].values.astype(float); dl = d['Low'].values.astype(float)
_, dp, dm = calc_i(dc, dh, dl)
d_dates = d.index; d_map = {}
for i, dd in enumerate(d_dates): d_map[dd] = {'pdi': dp[i], 'mdi': dm[i]}
def get_d(dt):
    nd = pd.Timestamp(dt).tz_localize(None).normalize()
    for dd in reversed(d_dates):
        if dd <= nd: return d_map.get(dd, {})
    return {}
d_pdi = np.full(len(c), np.nan); d_mdi = np.full(len(c), np.nan)
for i, dt in enumerate(h1.index):
    w = get_d(dt); d_pdi[i] = w.get('pdi', np.nan); d_mdi[i] = w.get('mdi', np.nan)

# ── CFD Config ──
MODAL = 1000.0
LEVERAGE = 100
LOT = 0.05                     # 0.05 lot = 5 oz
STOP_OUT = 30                  # Margin Level % for stop-out
CONTRACT_OZ = 100 * LOT        # oz per lot

def backtest(multi=False):
    label = "Multi TF" if multi else "Single TF"
    print(f"\n  {label}...")
    balance = MODAL; pos = None; eq = []; trades = []; mc_count = 0; n = len(c)
    
    for i in range(30, n):
        if i % 1000 == 0: print(f"    {i}/{n}...")
        
        current_price = c[i]
        
        # Check Margin Call / Stop-Out
        if pos is not None:
            used_margin = (pos['entry_price'] * CONTRACT_OZ) / LEVERAGE  # Fixed margin based on entry
            floating_pnl = (current_price - pos['entry_price']) * CONTRACT_OZ * pos['size']
            equity = balance + floating_pnl
            margin_level = (equity / used_margin) * 100
            
            # Stop-Out: close position if margin level below threshold
            if margin_level < STOP_OUT:
                used_margin_return = (pos['entry_price'] * CONTRACT_OZ) / LEVERAGE
                pnl = (current_price - pos['entry_price']) * CONTRACT_OZ * pos['size']
                balance += pnl + used_margin_return
                if balance < 0: balance = 0
                pct = (current_price / pos['entry_price'] - 1) * 100
                trades.append({'r': pct, 'ex': 'MC'})
                mc_count += 1; pos = None
                eq.append(balance)
                continue
            
            # Normal SL exit
            if current_price < pos['sl']:
                used_margin_return = (pos['entry_price'] * CONTRACT_OZ) / LEVERAGE
                pnl = (current_price - pos['entry_price']) * CONTRACT_OZ * pos['size']
                balance += pnl + used_margin_return
                if balance < 0: balance = 0
                pct = (current_price / pos['entry_price'] - 1) * 100
                trades.append({'r': pct, 'ex': 'SL'})
                pos = None
                eq.append(balance)
                continue
            
            # TP exit
            fl_pct = (current_price / pos['entry_price'] - 1) * 100
            if fl_pct > pos['tp']:
                used_margin_return = (pos['entry_price'] * CONTRACT_OZ) / LEVERAGE
                pnl = (current_price - pos['entry_price']) * CONTRACT_OZ * pos['size']
                balance += pnl + used_margin_return
                if balance < 0: balance = 0
                pct = (current_price / pos['entry_price'] - 1) * 100
                trades.append({'r': pct, 'ex': 'TP'})
                pos = None
                eq.append(balance)
                continue
        
        # Entry
        if pos is None and balance > 0:
            buy_ok = (h[i] > sl[i] and c[i] > sma20[i] and adx[i] > 20 
                      and pdi[i] > mdi[i] and pdi[i] > pdi[i-min(6,i)]
                      and not np.isnan(sl[i]) and sl[i] > 0)
            if multi: buy_ok = buy_ok and (d_pdi[i] > d_mdi[i])
            
            if buy_ok:
                used_margin = (c[i] * CONTRACT_OZ) / LEVERAGE
                if used_margin <= balance:  # Enough margin
                    sl_price = sl[i]
                    tp_pct = 0.4 * abs(c[i] - sl_price) / c[i] * 100
                    pos = {'entry_price': c[i], 'sl': sl_price, 'tp': tp_pct, 'size': LOT}
                    balance -= used_margin  # Margin locked
        
        if pos is None or True:
            equity = balance + (0 if pos is None else 
                (c[i] - pos['entry_price']) * CONTRACT_OZ * pos['size'])
            eq.append(equity)
    
    # Close final position
    if pos:
        used_margin_return = (pos['entry_price'] * CONTRACT_OZ) / LEVERAGE
        pnl = (c[-1] - pos['entry_price']) * CONTRACT_OZ * pos['size']
        balance += pnl + used_margin_return
        if balance < 0: balance = 0
    
    fe = eq[-1]; yrs = len(eq)/(252*7)
    r = {'ret': (fe-MODAL)/MODAL*100, 'n': len(trades), 'final': fe, 'mc': mc_count}
    if trades:
        dft = pd.DataFrame(trades)
        w = dft[dft['r']>0]; l_ = dft[dft['r']<=0]
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
        mc_df = dft[dft['ex']=='MC']
        r['mc_count'] = len(mc_df)
    return r

results = []
for mtf in [False, True]:
    r = backtest(mtf)
    results.append(r)
    lbl = "Multi TF" if mtf else "Single TF"
    print(f"\n{'='*40}")
    print(f"  XAU/USD CFD — {lbl}")
    print(f"{'='*40}")
    print(f"  Return    : {r['ret']:+.2f}%")
    print(f"  CAGR      : {r.get('cagr',0):+.2f}%/yr")
    print(f"  Sharpe    : {r.get('sharpe',0):.2f}")
    print(f"  Ulcer     : {r.get('ulcer',0):.2f}")
    print(f"  Max DD    : -{r.get('dd',0):.2f}%")
    print(f"  Trades    : {r['n']} | WR: {r.get('wr',0):.1f}%")
    print(f"  Margin Call: {r.get('mc_count',0)}x")
    print(f"  Final     : ${r['final']:.2f}")

print(f"\n\n{'='*55}")
print(f"  PERBANDINGAN XAU/USD — 0.1 lot + MC")
print(f"{'='*55}")
print(f"  {'Metrik':18s} {'Single TF':15s} {'Multi TF':15s}")
print('  ' + '-'*48)
for m in ['Return','CAGR','Sharpe','Ulcer','Max DD','Trades','WR','Margin Call']:
    k = m.lower().replace(' ','_')
    s = results[0].get(k, 0)
    m2 = results[1].get(k, 0)
    if m in ['Return','WR','Max DD','CAGR']:
        print(f"  {m:18s} {s:>8.2f}%      {m2:>8.2f}%")
    else:
        print(f"  {m:18s} {s:>8.2f}      {m2:>8.2f}")
