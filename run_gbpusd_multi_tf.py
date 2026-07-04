"""
Backtest — GBP/USD 30m, Single TF vs Multi TF
Multi-TF: existing conditions + Close>Basis_d & PDI_d>MDI_d (daily confirmation)
"""
import numpy as np, pandas as pd, yfinance as yf, warnings
warnings.filterwarnings("ignore")
from backtesting import Backtest, Strategy

print("Downloading GBP/USD 30m & Daily...")
# 30m
raw = yf.download("GBPUSD=X", period="1mo", interval="30m", progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
df = pd.DataFrame(index=raw.index)
for c in ['Open','High','Low','Close','Volume']:
    df[c] = raw[c].values.astype(float)

# Daily — for multi-TF confirmation
htf = yf.download("GBPUSD=X", period="6mo", interval="1d", progress=False)
if isinstance(htf.columns, pd.MultiIndex):
    htf.columns = htf.columns.get_level_values(0)
htf_c = htf['Close'].values.astype(float)
htf_h = htf['High'].values.astype(float)
htf_l = htf['Low'].values.astype(float)

# Daily indicators
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

htf_sma20 = pd.Series(htf_c).rolling(20).mean().values
htf_adx, htf_pdi, htf_mdi = calc_i(htf_c, htf_h, htf_l)

# Map daily to 30m bars by date
htf_dates = pd.Series(htf.index).dt.normalize()
htf_map = {}
for i, d in enumerate(htf_dates):
    htf_map[d] = {'sma': htf_sma20[i], 'pdi': htf_pdi[i], 'mdi': htf_mdi[i], 'close': htf_c[i]}

df['date'] = df.index.tz_localize(None).normalize()
df['htf_sma'] = df['date'].map(lambda d: htf_map.get(d, {}).get('sma', np.nan))
df['htf_pdi'] = df['date'].map(lambda d: htf_map.get(d, {}).get('pdi', np.nan))
df['htf_mdi'] = df['date'].map(lambda d: htf_map.get(d, {}).get('mdi', np.nan))
df['htf_close'] = df['date'].map(lambda d: htf_map.get(d, {}).get('close', np.nan))

df = df.dropna(subset=['htf_sma'])
print(f"  {len(df)} bars with daily data")

# ── Strategy Classes ──
class SingleTF(Strategy):
    def init(self):
        close = pd.Series(self.data.Close).values
        high = pd.Series(self.data.High).values
        low = pd.Series(self.data.Low).values
        self.sma = self.I(lambda: pd.Series(close).rolling(20).mean().values)
        tr = np.maximum(high-low, np.maximum(np.abs(high-np.roll(close,1)), np.abs(low-np.roll(close,1))))
        atr = pd.Series(tr).rolling(10).mean().values
        hh = pd.Series(high).rolling(28).max().values
        self.sl = self.I(lambda: hh - 2*atr*(2.8-1)/2.8)
        # ADX
        up = np.diff(high, prepend=high[0]); dn = np.diff(low, prepend=low[0])
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
            close = self.data.Close[-1]
            if (close > self.sma[-1] and self.data.Low[-1] > self.sl[-1]
                and self.adx[-1] > 20 and self.pdi[-1] > self.mdi[-1]):
                stop = float(self.sl[-1])
                if not np.isnan(stop) and stop > 0:
                    self.buy(sl=stop, tp=float(close + 0.4 * abs(close - stop)))

class MultiTF(Strategy):
    def init(self):
        close = pd.Series(self.data.Close).values
        high = pd.Series(self.data.High).values
        low = pd.Series(self.data.Low).values
        self.sma = self.I(lambda: pd.Series(close).rolling(20).mean().values)
        tr = np.maximum(high-low, np.maximum(np.abs(high-np.roll(close,1)), np.abs(low-np.roll(close,1))))
        atr = pd.Series(tr).rolling(10).mean().values
        hh = pd.Series(high).rolling(28).max().values
        self.sl = self.I(lambda: hh - 2*atr*(2.8-1)/2.8)
        up = np.diff(high, prepend=high[0]); dn = np.diff(low, prepend=low[0])
        pdm = np.where((up>dn)&(up>0), up, 0); mdm = np.where((dn>up)&(dn>0), dn, 0)
        atr_s = pd.Series(tr).rolling(14).mean().values
        sp = pd.Series(pdm).rolling(14).mean().values
        sm = pd.Series(mdm).rolling(14).mean().values
        pdi = np.where(atr_s>0, 100*sp/atr_s, 0); mdi = np.where(atr_s>0, 100*sm/atr_s, 0)
        dx = np.where((pdi+mdi)>0, 100*np.abs(pdi-mdi)/(pdi+mdi), 0)
        self.adx = self.I(lambda: pd.Series(dx).rolling(14).mean().values)
        self.pdi = self.I(lambda: pdi); self.mdi = self.I(lambda: mdi)
        # HTF (daily) indicators — passed as data columns
        self.htf_sma = self.I(lambda: df['htf_sma'].values)
        self.htf_pdi = self.I(lambda: df['htf_pdi'].values)
        self.htf_mdi = self.I(lambda: df['htf_mdi'].values)
    def next(self):
        if len(self.data) < 28: return
        if not self.position:
            close = self.data.Close[-1]
            # Current TF conditions
            cur_ok = (close > self.sma[-1] and self.data.Low[-1] > self.sl[-1]
                      and self.adx[-1] > 20 and self.pdi[-1] > self.mdi[-1])
            # Higher TF confirmation
            htf_ok = (close > self.htf_sma[-1] and self.htf_pdi[-1] > self.htf_mdi[-1])
            if cur_ok and htf_ok:
                stop = float(self.sl[-1])
                if not np.isnan(stop) and stop > 0:
                    self.buy(sl=stop, tp=float(close + 0.4 * abs(close - stop)))

# ── Run ──
for name, strat in [("Single TF (30m only)", SingleTF), ("Multi TF (30m + Daily)", MultiTF)]:
    bt = Backtest(df, strat, cash=10000, commission=.0005)
    r = bt.run()
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  Return    : {r['Return [%]']:+.2f}%")
    print(f"  Trades    : {r['# Trades']}")
    print(f"  Win Rate  : {r['Win Rate [%]']:.1f}%")
    print(f"  Max DD    : {r['Max. Drawdown [%]']:.2f}%")
    print(f"  Sharpe    : {r['Sharpe Ratio']:.2f}")
    print(f"  Best/Worst: +{r['Best Trade [%]']:.2f}% / {r['Worst Trade [%]']:.2f}%")
