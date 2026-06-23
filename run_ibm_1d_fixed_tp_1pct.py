"""Run Basis ADX 1D with FIXED 1.0% TP (min floating) and save HTML."""
import pandas as pd, numpy as np
from pathlib import Path
from backtesting import Backtest
from strategies.basis_adx_strategy import BasisAdxStrategy
import warnings; warnings.filterwarnings("ignore")

class BasisAdxFixedTP1Pct(BasisAdxStrategy):
    """Same as fixed-TP Basis ADX but with tp_min_profit_pct = 1.0%."""
    tp_min_profit_pct = 1.0   # <── 1% instead of 0.2%

    def next(self):
        idx = len(self.data) - 1
        close = float(self.data.Close[-1])
        low = float(self.data.Low[-1])
        high = float(self.data.High[-1])
        basis = float(self.basis[-1])
        sl = float(self.sl[-1])
        adx = float(self.adx_arr[idx])
        pdi = float(self.pdi_arr[idx])
        mdi = float(self.mdi_arr[idx])
        pdi_5ago = float(self.pdi_arr[idx - 5]) if idx >= 5 else 0.0
        sma200 = float(self.sma200[-1])
        uptrend = basis > sma200
        is_nan = np.isnan(adx) or np.isnan(pdi) or np.isnan(mdi) or np.isnan(sma200)

        # ── EXIT ──
        if self._entry_sl is not None and self._entry_price is not None:
            if close < self._entry_sl:
                self.position.close()
                self._entry_sl = None
                self._entry_price = None
                return
            if self._entry_price > 0:
                floating_pct = ((close - self._entry_price) / self._entry_price) * 100.0
                if floating_pct > self.tp_min_profit_pct and close < sl:
                    self.position.close()
                    self._entry_sl = None
                    self._entry_price = None
                    return
            return

        # ── ENTRY ──
        if (low > sl and close > basis and not is_nan and adx > 20.0 and uptrend
                and pdi > mdi and pdi > pdi_5ago):
            stop_dist = abs(close - sl)
            if stop_dist > 0:
                risk_amount = self.equity * (self.risk_pct / 100.0)
                size = int(risk_amount / stop_dist)
                max_by_cash = int((self.equity * 0.95) / close)
                if max_by_cash >= 1:
                    size = max(1, size)
                size = min(size, max_by_cash)
            else:
                size = 0
            if size > 0:
                self.buy(size=size)
                self._entry_sl = sl
                self._entry_price = close

BASE = Path(r"C:\Users\satri\code\backtester")
df = pd.read_csv(BASE/"IBM_1d.txt", parse_dates=["DateTime"])
df = df.rename(columns={"DateTime":"Date"}).set_index("Date").sort_index()
df = df.dropna(subset=["Open","High","Low","Close"])
df = df[df["Volume"]>0]

bt = Backtest(df, BasisAdxFixedTP1Pct, cash=100_000, commission=0.001, finalize_trades=True)
stats = bt.run()

print(f"\n{'='*55}")
print(f"  IBM 1D — Basis ADX (fixed 1.0% TP / min floating)")
print(f"{'='*55}")
print(f"  Return       : {stats['Return [%]']:.2f}%")
print(f"  Buy & Hold   : {stats['Buy & Hold Return [%]']:.2f}%")
print(f"  Equity Final : ${stats['Equity Final [$]']:,.2f}")
print(f"  Sharpe       : {stats['Sharpe Ratio']:.2f}")
print(f"  Max DD       : {stats['Max. Drawdown [%]']:.2f}%")
print(f"  # Trades     : {stats['# Trades']}")
print(f"  Win Rate     : {stats.get('Win Rate [%]', 0):.1f}%")
print(f"  Best Trade   : {stats.get('Best Trade [%]', 0):.2f}%")
print(f"  Worst Trade  : {stats.get('Worst Trade [%]', 0):.2f}%")
print(f"  Avg Trade    : {stats.get('Avg Trade [%]', 0):.2f}%")

report_path = BASE/"reports"/"IBM_basis_adx_fixed_1pct_1d.html"
bt.plot(filename=str(report_path), open_browser=False)
print(f"\n  Report       : {report_path}")
print("✅ DONE")
