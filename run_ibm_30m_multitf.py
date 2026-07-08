"""
Backtest — IBM 30 menit (1998-2026)
Single TF (30m) vs Multi TF (30m + Daily PDI>MDI)
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from backtesting import Backtest, Strategy

print("Loading IBM 30m data...")
df = pd.read_csv(r'C:\Users\Acer\AppData\Local\hermes\cache\documents\doc_dbdfa09eafdf_IBM_30min.txt',
    parse_dates=['DateTime'], dayfirst=True)
df = df.rename(columns={'DateTime':'datetime','Open':'Open','High':'High','Low':'Low','Close':'Close','Volume':'Volume'})
df = df.set_index('datetime').sort_index()
df.index = pd.to_datetime(df.index)
print(f"  Bars: {len(df)} | {df.index[0]} → {df.index[-1]}")

# ── Indicators ──
c, h, l = df['Close'].values.astype(float), df['High'].values.astype(float), df['Low'].values.astype(float)

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

df['SMA20'] = pd.Series(c).rolling(20).mean()
tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
df['SL'] = pd.Series(c).rolling(28).max().values - 2 * pd.Series(tr).rolling(10).mean().values * (2.8-1)/2.8
adx, pdi, mdi = calc_i(c, h, l)
df['ADX'], df['PDI'], df['MDI'] = adx, pdi, mdi

# ── Daily HTF from resampled 30m data ──
daily = df.resample('D').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
dc = daily['Close'].values.astype(float); dh = daily['High'].values.astype(float); dl = daily['Low'].values.astype(float)
_, dp, dm = calc_i(dc, dh, dl)

# Map daily PDI/MDI to 30m bars
d_dates = daily.index; d_map = {}
for i, dd in enumerate(d_dates): d_map[dd] = {'pdi': dp[i], 'mdi': dm[i]}
def get_d(dt):
    for dd in reversed(d_dates):
        if dd <= dt: return d_map.get(dd, {})
    return {}
hp = np.full(len(df), np.nan); hm = np.full(len(df), np.nan)
for i, dt in enumerate(df.index):
    w = get_d(dt); hp[i] = w.get('pdi', np.nan); hm[i] = w.get('mdi', np.nan)
df['D_PDI'], df['D_MDI'] = hp, hm

df = df.dropna()
print(f"  After indicators: {len(df)} bars")

# ── Strategy 1: Single TF ──
class SingleTF(Strategy):
    def init(self):
        self.sma = self.I(lambda: df['SMA20'].values)
        self.sl = self.I(lambda: df['SL'].values)
        self.adx = self.I(lambda: df['ADX'].values)
        self.pdi = self.I(lambda: df['PDI'].values)
        self.mdi = self.I(lambda: df['MDI'].values)
    def next(self):
        if len(self.data) < 30: return
        if not self.position:
            if (self.data.Close[-1] > self.sma[-1] and self.data.Low[-1] > self.sl[-1]
                and self.adx[-1] > 20 and self.pdi[-1] > self.mdi[-1]
                and self.pdi[-1] > self.pdi[-min(6, len(self.data))]):
                stop = float(self.sl[-1])
                if not np.isnan(stop) and stop > 0:
                    self.buy(sl=stop, tp=float(self.data.Close[-1] + 0.4*abs(self.data.Close[-1]-stop)))

# ── Strategy 2: Multi TF ──
class MultiTF(Strategy):
    def init(self):
        self.sma = self.I(lambda: df['SMA20'].values)
        self.sl = self.I(lambda: df['SL'].values)
        self.adx = self.I(lambda: df['ADX'].values)
        self.pdi = self.I(lambda: df['PDI'].values)
        self.mdi = self.I(lambda: df['MDI'].values)
        self.dpdi = self.I(lambda: df['D_PDI'].values)
        self.dmdi = self.I(lambda: df['D_MDI'].values)
    def next(self):
        if len(self.data) < 30: return
        if not self.position:
            cur_ok = (self.data.Close[-1] > self.sma[-1] and self.data.Low[-1] > self.sl[-1]
                and self.adx[-1] > 20 and self.pdi[-1] > self.mdi[-1]
                and self.pdi[-1] > self.pdi[-min(6, len(self.data))])
            htf_ok = (self.dpdi[-1] > self.dmdi[-1])
            if cur_ok and htf_ok:
                stop = float(self.sl[-1])
                if not np.isnan(stop) and stop > 0:
                    self.buy(sl=stop, tp=float(self.data.Close[-1] + 0.4*abs(self.data.Close[-1]-stop)))

# ── Run ──
for name, strat in [("Single TF (30m)", SingleTF), ("Multi TF (30m+Daily)", MultiTF)]:
    bt = Backtest(df, strat, cash=100000, commission=.001)
    r = bt.run()
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    for k in ['Return [%]','# Trades','Win Rate [%]','Max. Drawdown [%]','Sharpe Ratio','Best Trade [%]','Worst Trade [%]']:
        print(f"  {k}: {r[k]}")
