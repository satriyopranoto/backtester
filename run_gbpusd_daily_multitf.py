"""
Backtest — GBP/USD Daily, Single TF vs Multi TF (Daily+Weekly).
Multi-TF: existing daily conditions + Close>Basis_w & PDI_w>MDI_w (weekly confirmation)
"""
import numpy as np, pandas as pd, yfinance as yf, warnings
warnings.filterwarnings("ignore")
from backtesting import Backtest, Strategy

print("Downloading GBP/USD Daily & Weekly...")
# Daily
raw = yf.download("GBPUSD=X", period="5y", interval="1d", progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
df = pd.DataFrame(index=raw.index)
for c in ['Open','High','Low','Close','Volume']:
    df[c] = raw[c].values.astype(float)
print(f"  Daily: {len(df)} bars | {df.index[0].date()} → {df.index[-1].date()}")

# Weekly
wk = yf.download("GBPUSD=X", period="5y", interval="1wk", progress=False)
if isinstance(wk.columns, pd.MultiIndex):
    wk.columns = wk.columns.get_level_values(0)
print(f"  Weekly: {len(wk)} bars | {wk.index[0].date()} → {wk.index[-1].date()}")

# Weekly indicators
wk_c = wk['Close'].values.astype(float)
wk_h = wk['High'].values.astype(float)
wk_l = wk['Low'].values.astype(float)
wk_sma20 = pd.Series(wk_c).rolling(20).mean().values

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

wk_adx, wk_pdi, wk_mdi = calc_i(wk_c, wk_h, wk_l)

# Map weekly to daily bars
# Each daily bar gets the most recent weekly data (prior week)
wk_dates = wk.index
wk_map = {}
for i, wd in enumerate(wk_dates):
    wk_map[wd] = {'sma': wk_sma20[i], 'pdi': wk_pdi[i], 'mdi': wk_mdi[i], 'close': wk_c[i]}

# For each daily bar, find the last weekly bar BEFORE or ON that date
df['date'] = df.index.normalize()
def get_weekly(d):
    # Find the latest weekly bar <= this date
    for wd in reversed(wk_dates):
        if wd <= d:
            return wk_map.get(wd, {})
    return {}

df['wk_sma'] = df['date'].apply(lambda d: get_weekly(d).get('sma', np.nan))
df['wk_pdi'] = df['date'].apply(lambda d: get_weekly(d).get('pdi', np.nan))
df['wk_mdi'] = df['date'].apply(lambda d: get_weekly(d).get('mdi', np.nan))

df = df.dropna(subset=['wk_sma'])
print(f"  After merge: {len(df)} bars")

# Drop early bars before weekly SMA20 kicks in (first 20 weeks ~140 days)
min_idx = np.where(~np.isnan(wk_sma20))[0]
if len(min_idx) > 0:
    first_valid_wk = wk_dates[min_idx[0]]
    df = df[df['date'] >= first_valid_wk]
print(f"  After weekly SMA20 warmup: {len(df)} bars")
print()

# ── Strategy 1: Single TF (Daily only) ──
class SingleTF(Strategy):
    def init(self):
        c = pd.Series(self.data.Close).values; h = pd.Series(self.data.High).values; l = pd.Series(self.data.Low).values
        self.sma = self.I(lambda: pd.Series(c).rolling(20).mean().values)
        tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
        atr = pd.Series(tr).rolling(10).mean().values
        self.sl = self.I(lambda: pd.Series(c).rolling(28).max().values - 2*atr*(2.8-1)/2.8)
        up = np.diff(h, prepend=h[0]); dn = np.diff(l, prepend=l[0])
        pdm = np.where((up>dn)&(up>0), up, 0); mdm = np.where((dn>up)&(dn>0), dn, 0)
        atr_s = pd.Series(tr).rolling(14).mean().values
        sp = pd.Series(pdm).rolling(14).mean().values
        sm = pd.Series(mdm).rolling(14).mean().values
        pdi = np.where(atr_s>0, 100*sp/atr_s, 0); mdi = np.where(atr_s>0, 100*sm/atr_s, 0)
        dx = np.where((pdi+mdi)>0, 100*np.abs(pdi-mdi)/(pdi+mdi), 0)
        self.adx = self.I(lambda: pd.Series(dx).rolling(14).mean().values)
        self.pdi = self.I(lambda: pdi); self.mdi = self.I(lambda: mdi)
    def next(self):
        if len(self.data) < 28: return
        if not self.position:
            if (self.data.Close[-1] > self.sma[-1] and 
                self.data.Low[-1] > self.sl[-1] and
                self.adx[-1] > 20 and self.pdi[-1] > self.mdi[-1]):
                stop = float(self.sl[-1])
                if not np.isnan(stop) and stop > 0:
                    self.buy(sl=stop, tp=float(self.data.Close[-1] + 0.4*abs(self.data.Close[-1]-stop)))

# ── Strategy 2: Multi TF (Daily + Weekly) ──
class MultiTF(Strategy):
    def init(self):
        c = pd.Series(self.data.Close).values; h = pd.Series(self.data.High).values; l = pd.Series(self.data.Low).values
        self.sma = self.I(lambda: pd.Series(c).rolling(20).mean().values)
        tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
        atr = pd.Series(tr).rolling(10).mean().values
        self.sl = self.I(lambda: pd.Series(c).rolling(28).max().values - 2*atr*(2.8-1)/2.8)
        up = np.diff(h, prepend=h[0]); dn = np.diff(l, prepend=l[0])
        pdm = np.where((up>dn)&(up>0), up, 0); mdm = np.where((dn>up)&(dn>0), dn, 0)
        atr_s = pd.Series(tr).rolling(14).mean().values
        sp = pd.Series(pdm).rolling(14).mean().values
        sm = pd.Series(mdm).rolling(14).mean().values
        pdi = np.where(atr_s>0, 100*sp/atr_s, 0); mdi = np.where(atr_s>0, 100*sm/atr_s, 0)
        dx = np.where((pdi+mdi)>0, 100*np.abs(pdi-mdi)/(pdi+mdi), 0)
        self.adx = self.I(lambda: pd.Series(dx).rolling(14).mean().values)
        self.pdi = self.I(lambda: pdi); self.mdi = self.I(lambda: mdi)
        # Weekly HTF
        self.wk_sma = self.I(lambda: df['wk_sma'].values)
        self.wk_pdi = self.I(lambda: df['wk_pdi'].values)
        self.wk_mdi = self.I(lambda: df['wk_mdi'].values)
    def next(self):
        if len(self.data) < 28: return
        if not self.position:
            cur_ok = (self.data.Close[-1] > self.sma[-1] and 
                      self.data.Low[-1] > self.sl[-1] and
                      self.adx[-1] > 20 and self.pdi[-1] > self.mdi[-1])
            htf_ok = (self.data.Close[-1] > self.wk_sma[-1] and 
                      self.wk_pdi[-1] > self.wk_mdi[-1])
            if cur_ok and htf_ok:
                stop = float(self.sl[-1])
                if not np.isnan(stop) and stop > 0:
                    self.buy(sl=stop, tp=float(self.data.Close[-1] + 0.4*abs(self.data.Close[-1]-stop)))

# ── Run ──
for name, strat in [("Single TF (Daily only)", SingleTF), ("Multi TF (Daily+Weekly)", MultiTF)]:
    try:
        bt = Backtest(df, strat, cash=100000, commission=.001)
        r = bt.run()
        print(f"\n{'='*40}")
        print(f"  {name}")
        print(f"{'='*40}")
        for k in ['Return [%]','# Trades','Win Rate [%]','Max. Drawdown [%]','Sharpe Ratio','Best Trade [%]','Worst Trade [%]']:
            print(f"  {k}: {r[k]}")
    except Exception as e:
        print(f"\n  {name} ERROR: {e}")
