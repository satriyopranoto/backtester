"""
Backtest — GBP/USD 30m, 1 month last.
"""
import numpy as np, pandas as pd, yfinance as yf, warnings
warnings.filterwarnings("ignore")
from backtesting import Backtest, Strategy

# Download
print("Downloading GBP/USD 30m...")
raw = yf.download("GBPUSD=X", period="1mo", interval="30m", progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

# Use only bare OHLCV columns — let backtesting handle indicators
df = pd.DataFrame(index=raw.index)
for c in ['Open','High','Low','Close','Volume']:
    df[c] = raw[c].values.astype(float)

print(f"  {len(df)} bars: {df.index[0]} → {df.index[-1]}")

# ── Strategy 1: Simple Basis+ADX ──
class BasisAdx(Strategy):
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        # SMA20
        self.sma = self.I(lambda: close.rolling(20).mean().values, name='SMA20')
        
        # ATR-based SL
        tr = np.maximum(high.values-low.values, 
            np.maximum(np.abs(high.values-np.roll(close.values,1)),
                       np.abs(low.values-np.roll(close.values,1))))
        atr = pd.Series(tr).rolling(10).mean().values
        highest = pd.Series(high.values).rolling(28).max().values
        self.sl = self.I(lambda: (highest - 2 * atr * (2.8-1)/2.8), name='SL')
        
        # ADX
        p = 14
        up = np.diff(high.values, prepend=high.values[0])
        dn = np.diff(low.values, prepend=low.values[0])
        pdm = np.where((up>dn)&(up>0), up, 0)
        mdm = np.where((dn>up)&(dn>0), dn, 0)
        atr_s = pd.Series(tr).rolling(p).mean().values
        sp = pd.Series(pdm).rolling(p).mean().values
        sm = pd.Series(mdm).rolling(p).mean().values
        pdi = np.where(atr_s>0, 100*sp/atr_s, 0)
        mdi = np.where(atr_s>0, 100*sm/atr_s, 0)
        dx = np.where((pdi+mdi)>0, 100*np.abs(pdi-mdi)/(pdi+mdi), 0)
        self.adx = self.I(lambda: pd.Series(dx).rolling(p).mean().values, name='ADX')
        self.pdi = self.I(lambda: pdi, name='PDI')
        self.mdi = self.I(lambda: mdi, name='MDI')
        
    def next(self):
        if len(self.data) < 28: return
        if not self.position:
            cond = (self.data.Close[-1] > self.sma[-1] and 
                    self.data.Low[-1] > self.sl[-1] and
                    self.adx[-1] > 20 and self.pdi[-1] > self.mdi[-1])
            if cond and not np.isnan(self.sl[-1]) and self.sl[-1] > 0:
                stop = float(self.sl[-1])
                self.buy(sl=stop, tp=float(self.data.Close[-1] + 0.4 * abs(self.data.Close[-1] - stop)))

# ── Strategy 2: Mean Reversion ──
class MeanRev(Strategy):
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        
        delta = np.diff(close.values, prepend=close.values[0])
        gain = np.where(delta>0, delta, 0)
        loss = np.where(delta<0, -delta, 0)
        avg_g = pd.Series(gain).rolling(14).mean().values
        avg_l = pd.Series(loss).rolling(14).mean().values
        rs = np.divide(avg_g, avg_l, out=np.full_like(avg_g, np.inf), where=avg_l!=0)
        self.rsi = self.I(lambda: 100 - (100/(1+rs)), name='RSI')
        
        sma = close.rolling(20).mean().values
        std = close.rolling(20).std().values
        self.lbb = self.I(lambda: sma - 2*std, name='LBB')
        self.ubb = self.I(lambda: sma + 2*std, name='UBB')
        
    def next(self):
        if len(self.data) < 25: return
        if not self.position:
            if self.rsi[-1] < 30 and self.data.Close[-1] < self.lbb[-1]:
                self.buy(sl=self.data.Close[-1]*0.995, tp=self.data.Close[-1]*1.005)
        else:
            if self.rsi[-1] > 70 and self.data.High[-1] > self.ubb[-1]:
                self.position.close()

# ── Run ──
for name, strat in [("Basis+ADX", BasisAdx), ("Mean Reversion", MeanRev)]:
    try:
        bt = Backtest(df, strat, cash=10000, commission=.0005)
        r = bt.run()
        print(f"\n{'='*35}")
        print(f"  {name}")
        print(f"{'='*35}")
        for k in ['Return [%]','# Trades','Win Rate [%]','Max. Drawdown [%]','Sharpe Ratio']:
            print(f"  {k}: {r[k]}")
    except Exception as e:
        print(f"\n  {name} ERROR: {e}")
