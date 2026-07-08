"""
USD/CHF 1h — Mean Reversion + CFD (0.05 lot)
Entry: RSI < 30 + Close < Lower BB
Exit:  RSI > 70 + High > Upper BB or SL
"""
import numpy as np, pandas as pd, yfinance as yf, warnings
warnings.filterwarnings('ignore')

print("Loading USD/CHF 1h...")
t = 'USDCHF=X'
h1 = yf.download(t, period='1y', interval='1h', progress=False)
if isinstance(h1.columns, pd.MultiIndex): h1.columns = h1.columns.get_level_values(0)
c = h1['Close'].values.astype(float); h = h1['High'].values.astype(float); l = h1['Low'].values.astype(float)
print(f"  {len(h1)} bars | {h1.index[0]} → {h1.index[-1]}")

# Indicators
sma20 = pd.Series(c).rolling(20).mean().values
std20 = pd.Series(c).rolling(20).std().values
upper_bb = sma20 + 2 * std20
lower_bb = sma20 - 2 * std20

delta = np.diff(c, prepend=c[0])
gain = np.where(delta>0, delta, 0); loss = np.where(delta<0, -delta, 0)
avg_g = pd.Series(gain).rolling(14).mean().values
avg_l = pd.Series(loss).rolling(14).mean().values
rs = np.divide(avg_g, avg_l, out=np.full_like(avg_g, np.inf), where=avg_l!=0)
rsi = 100 - (100 / (1 + rs))

tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
atr = pd.Series(tr).rolling(14).mean().values

# CFD Config
MODAL = 1000.0
LEVERAGE = 100
LOT = 0.05
STOP_OUT = 30
CONTRACT = 100000 * LOT  # 5000 units for 0.05 lot
atr_now = pd.Series(tr).rolling(10).mean().iloc[-1]
price_now = c[-1]

print(f"\nModal: $1,000 | Leverage: 1:{LEVERAGE} | Lot: {LOT}")
print(f"Contract: {CONTRACT:.0f} units | Nilai per pip: ${CONTRACT*0.0001:.2f}")
print(f"Harga: {price_now:.5f} | ATR: {atr_now*10000:.0f} pips")

def backtest():
    balance = MODAL; pos = None; eq = []; trades = []; mc = 0
    n = len(c)
    for i in range(30, n):
        if i % 1000 == 0: print(f"    {i}/{n}...")
        p = c[i]
        
        # MC check & exits
        if pos:
            um = (pos['ep'] * CONTRACT) / LEVERAGE
            fp = (p - pos['ep']) * CONTRACT * pos['sz']
            eq_ = balance + fp
            ml = (eq_ / um) * 100
            
            if ml < STOP_OUT:
                bal_ret = (pos['ep'] * CONTRACT * pos['sz']) / LEVERAGE
                balance += fp + bal_ret
                if balance < 0: balance = 0
                trades.append({'r': (p/pos['ep']-1)*100, 'ex': 'MC'}); mc+=1; pos=None
                eq.append(balance); continue
            
            # SL (fixed 2% of price? actually use ATR-based)
            if p < pos['sl']:
                bal_ret = (pos['ep'] * CONTRACT * pos['sz']) / LEVERAGE
                balance += fp + bal_ret
                if balance < 0: balance = 0
                trades.append({'r': (p/pos['ep']-1)*100, 'ex': 'SL'}); pos=None
                eq.append(balance); continue
            
            # Mean Reversion TP: RSI > 70 + price > Upper BB
            if not np.isnan(rsi[i]) and not np.isnan(upper_bb[i]) and rsi[i] > 70 and p > upper_bb[i]:
                bal_ret = (pos['ep'] * CONTRACT * pos['sz']) / LEVERAGE
                balance += fp + bal_ret
                if balance < 0: balance = 0
                trades.append({'r': (p/pos['ep']-1)*100, 'ex': 'MR_TP'}); pos=None
                eq.append(balance); continue
        
        # Entry: RSI < 30 + Close < Lower BB
        if pos is None and balance > 0 and not (np.isnan(rsi[i]) or np.isnan(lower_bb[i])):
            if rsi[i] < 30 and p < lower_bb[i]:
                um = (p * CONTRACT) / LEVERAGE
                if um <= balance:
                    sl_price = p * 0.995  # 0.5% SL
                    pos = {'ep': p, 'sl': sl_price, 'sz': LOT}
                    balance -= um
        
        eq_e = balance + (0 if pos is None else (p - pos['ep']) * CONTRACT * pos['sz'])
        eq.append(eq_e)
    
    if pos:
        bal_ret = (pos['ep'] * CONTRACT * pos['sz']) / LEVERAGE
        fp = (c[-1] - pos['ep']) * CONTRACT * pos['sz']
        balance += fp + bal_ret
        if balance < 0: balance = 0
    
    fe = eq[-1]; yrs = len(eq)/(252*7)
    r = {'ret': (fe-MODAL)/MODAL*100, 'n': len(trades), 'final': fe, 'mc': mc}
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
    return r

print(f"\nRunning Mean Reversion backtest...")
r = backtest()

print(f"\n{'='*40}")
print(f"  USD/CHF — Mean Reversion")
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
