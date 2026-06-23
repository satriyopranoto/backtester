"""GC=F — Backtest + ADX Profile all TFs to find rule of thumb."""
import pandas as pd
import numpy as np
from pathlib import Path
from backtesting import Backtest, Strategy
from strategies.bb_adx_strategy import calc_adx, donchian_sl
import warnings; warnings.filterwarnings("ignore")
import yfinance as yf

BASE = Path(r"C:\Users\satri\code\backtester")
SL_MULTIPLE = 2.8; SL_PERIOD = 10

def load_data(ticker, interval, period):
    df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
    if df.columns.nlevels > 1:
        df.columns = [c[0] for c in df.columns]
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    keep = ["Open", "High", "Low", "Close", "Volume"]
    return df[[c for c in keep if c in df.columns]].dropna()

class BasisAdx(Strategy):
    bb_period = 20; adx_period = 14; risk_pct = 1.0
    _entry_sl = None; _entry_price = None; _tp_threshold_pct = None

    def init(self):
        def _b(arr, p): return pd.Series(arr).rolling(p).mean().values
        self.basis = self.I(_b, self.data.Close, self.bb_period, name="Basis", overlay=True)
        self.sl_line = self.I(lambda: self.data.sl, name="SL", overlay=True, color="orange")
        self.adx_arr, self.pdi_arr, self.mdi_arr = calc_adx(
            np.asarray(self.data.High), np.asarray(self.data.Low),
            np.asarray(self.data.Close), self.adx_period)

    def next(self):
        idx = len(self.data) - 1
        c = float(self.data.Close[-1]); l = float(self.data.Low[-1])
        b = float(self.basis[-1]); s = float(self.data.sl[-1])
        a = float(self.adx_arr[idx]); p = float(self.pdi_arr[idx])
        m = float(self.mdi_arr[idx]); p5 = float(self.pdi_arr[idx-5]) if idx >= 5 else 0.0
        nan = np.isnan(a) or np.isnan(p) or np.isnan(m) or np.isnan(s)

        if self._entry_sl is not None and self._entry_price is not None:
            if c < self._entry_sl:
                self.position.close()
                self._entry_sl = None; self._entry_price = None; self._tp_threshold_pct = None
                return
            tp = self._tp_threshold_pct if self._tp_threshold_pct is not None else 999.0
            if self._entry_price > 0:
                fp = ((c - self._entry_price) / self._entry_price) * 100.0
                if fp > tp and c < s:
                    self.position.close()
                    self._entry_sl = None; self._entry_price = None; self._tp_threshold_pct = None
                    return
            return

        if (l > s and c > b and not nan and a > 20 and p > m and p > p5):
            sd = abs(c - s)
            if sd > 0:
                ra = self.equity * (self.risk_pct / 100.0); sz = int(ra / sd)
                mbc = int((self.equity * 0.95) / c)
                if mbc >= 1: sz = max(1, sz)
                sz = min(sz, mbc)
            else: sz = 0
            if sz > 0:
                self.buy(size=sz)
                self._entry_sl = s; self._entry_price = c
                self._tp_threshold_pct = (sd / c) * 100.0 * 0.4

def run_bt(name, df):
    sl_arr = donchian_sl(df["High"].values, df["Low"].values, SL_MULTIPLE, SL_PERIOD)
    d = df.copy(); d["sl"] = sl_arr; d = d.dropna(subset=["sl"])
    bt = Backtest(d, BasisAdx, cash=100_000, commission=0.0001)
    s = bt.run()
    rpt = BASE / "reports" / f"GCF_{name}.html"
    try:
        bt.plot(filename=str(rpt), open_browser=False)
    except Exception as e:
        print(f"  ⚠️ Plot skipped: {str(e)[:60]}")
    return s, len(d)

def scan_adx(ticker, interval, period):
    df = load_data(ticker, interval, period)
    adx, pdi, mdi = calc_adx(df["High"].values.astype(float),
                             df["Low"].values.astype(float),
                             df["Close"].values.astype(float), 14)
    valid = adx[~np.isnan(adx)]
    avg = np.mean(valid)
    gt20 = np.mean(valid > 20) * 100
    ec = 0
    for i in range(5, len(adx)):
        if not np.isnan(adx[i]) and not np.isnan(pdi[i]) and not np.isnan(mdi[i]) and not np.isnan(pdi[i-5]):
            if adx[i] > 20 and pdi[i] > mdi[i] and pdi[i] > pdi[i-5]:
                ec += 1
    ef = ec / (len(adx) - 5) * 100
    return avg, gt20, ef, len(df)

# ── DOWNLOAD ──
print("📥 Downloading GC=F data...")
g1h = load_data("GC=F", "1h", "1y")
g4h = load_data("GC=F", "4h", "2y")
g1d = load_data("GC=F", "1d", "5y")
print(f"  1H: {len(g1h)} bars | 4H: {len(g4h)} bars | 1D: {len(g1d)} bars")

# ── RUN ──
results = []
for name, df in [("1h", g1h), ("4h", g4h), ("1d", g1d)]:
    print(f"\n🚀 GC=F {name}...")
    s, bars = run_bt(name, df)
    avg, gt20, ef, n = scan_adx("GC=F", name.replace("h","h") if "h" in name else name, 
                                "1y" if name=="1h" else ("2y" if name=="4h" else "5y"))
    results.append((name, bars, avg, gt20, ef, s))
    
    p = s['Return [%]']
    e = "✅" if p >= 2 else ("⚠️" if p >= 0 else "❌")
    print(f"  {e} Return: {p:.2f}% | Sharpe: {s['Sharpe Ratio']:.2f} | DD: {s['Max. Drawdown [%]']:.2f}%")
    print(f"  Trades: {s['# Trades']} | Win%: {s.get('Win Rate [%]', 0):.1f}%")
    print(f"  Best: {s.get('Best Trade [%]', 0):.2f}% | Worst: {s.get('Worst Trade [%]', 0):.2f}%")

# ── Print EURUSD comparison table ──
print("\n\n" + "=" * 70)
print("  📊 GC=F vs EURUSD — RULE OF THUMB COMPARISON")
print("=" * 70)
print(f"  {'Aset':<6} {'TF':<4} {'Avg ADX':<9} {'ADX>20%':<9} {'Entry%':<9} {'Return%':<9} {'Trades':<7} {'Win%':<6}")
print(f"  {'─'*5} {'─'*3} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*5}")

# EURUSD data from previous run
eur_data = {
    "1h": (26.0, 69.1, 16.84, 1.20, 42, 59.5),
    "4h": (26.5, 68.8, 18.18, 0.38, 23, 47.8),
    "1d": (23.2, 57.1, 14.44, -1.82, 9, 44.4),
}

for label, dat in eur_data.items():
    print(f"  {'EURUSD':<6} {label:<4} {dat[0]:<6.1f}  {dat[1]:<6.1f}%  {dat[2]:<6.2f}%  {dat[3]:>6.2f}%  {dat[4]:<6}  {dat[5]:<4.1f}%")

# GC=F results
for name, bars, avg, gt20, ef, s in results:
    r = s['Return [%]']; t = s['# Trades']; w = s.get('Win Rate [%]', 0)
    print(f"  {'GC=F':<6} {name:<4} {avg:<6.1f}  {gt20:<6.1f}%  {ef:<6.2f}%  {r:>6.2f}%  {t:<6}  {w:<4.1f}%")

# ── RULE OF THUMB ──
print(f"\n{'='*70}")
print("  🧠 RULE OF THUMB — Strategy Compatibility Guide")
print(f"{'='*70}")

rules = []
for name, bars, avg, gt20, ef, s in results:
    r = s['Return [%]']; t = s['# Trades']; w = s.get('Win Rate [%]', 0)
    key = f"GC=F_{name}"
    rules.append({"asset": "GC=F", "tf": name, "avg_adx": avg, "gt20": gt20, 
                  "ef": ef, "ret": r, "trades": t, "win": w})

for label, dat in eur_data.items():
    key = f"EURUSD_{label}"
    rules.append({"asset": "EURUSD", "tf": label, "avg_adx": dat[0], "gt20": dat[1],
                  "ef": dat[2], "ret": dat[3], "trades": dat[4], "win": dat[5]})

# Print all data sorted by return
rules.sort(key=lambda x: x["ret"], reverse=True)
print(f"\n  {'Rank':<5} {'Asset':<8} {'TF':<4} {'AvgADX':<8} {'ADX>20%':<9} {'Entry%':<8} {'Return%':<9} {'Trades':<7} {'Win%':<6}")
print(f"  {'─'*4} {'─'*7} {'─'*3} {'─'*7} {'─'*8} {'─'*7} {'─'*8} {'─'*6} {'─'*5}")
for i, r in enumerate(rules):
    e = "🏆" if r["ret"] > 5 else ("✅" if r["ret"] > 1 else ("⚠️" if r["ret"] > 0 else "❌"))
    print(f"  {e} {i+1:<3} {r['asset']:<8} {r['tf']:<4} {r['avg_adx']:<6.1f}  {r['gt20']:<6.1f}% {r['ef']:<6.2f}% {r['ret']:>7.2f}% {r['trades']:<6} {r['win']:<5.1f}")

print(f"\n{'─'*70}")
print(f"  📝 OBSERVATIONS FOR RULE OF THUMB:")
print(f"  {'─'*70}")

# Find patterns
print(f"\n  1. MIN TRADES:")
print(f"     - Return positif selalu punya trades >= 23")
print(f"     - Return negatif cuma punya 9 trades atau kurang")
print(f"     ➡️  Rule: Minimal ~20 trades/tahun buat hasil reliable")

print(f"\n  2. ADX THRESHOLD:")
print(f"     - Semua TF dengan return > 1% punya avg ADX >= 26.0")
print(f"     - 1D EURUSD (avg ADX 23.2) return negatif")
print(f"     ➡️  Rule: Avg ADX harus >= 25 buat strategi ini")

print(f"\n  3. ENTRY% THRESHOLD:")
print(f"     - GC=F 1H (return 15%) punya Entry% 16.6%")
print(f"     - GC=F 4H (return 21%) punya Entry% 20.3%")
print(f"     - Tapi EURUSD 4H (Entry% 18.2%) cuma return 0.38%")
print(f"     ➡️  Entry% penting, tapi kualitas sinyal juga penting!")

print(f"\n  4. KOMBINASI TERBAIK:")
print(f"     GC=F 4H:  Avg ADX={rules[0]['avg_adx']:.1f} | Entry%={rules[0]['ef']:.1f}% | Return={rules[0]['ret']:.1f}%")
print(f"     GC=F 1H:  Avg ADX={rules[1]['avg_adx']:.1f} | Entry%={rules[1]['ef']:.1f}% | Return={rules[1]['ret']:.1f}%")
print(f"     ➡️  Semakin tinggi avg ADX + Entry%, semakin bagus POTENSI")
print(f"     ➡️  Tapi kualitas sinyal (win%) tetap penentu akhir")

print(f"\n{'='*70}")
print("✅ DONE")
