"""
Backtest Basis+ADX (Trend) — TSLA Daily, last 5 years. Comparison vs Mean Rev ADX.
"""
import numpy as np, pandas as pd, yfinance as yf, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from backtesting import Backtest
from strategies.basis_adx_strategy import BasisAdxStrategy

print("=" * 60)
print("  Basis+ADX (Trend) — TSLA Daily (5 tahun)")
print("  (versus Mean Rev ADX: +2.26%)")
print("=" * 60)

ticker = "TSLA"
data = yf.download(ticker, period="5y", interval="1d", auto_adjust=True)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data = data.dropna(subset=["Open","High","Low","Close"])
print(f"  Bars: {len(data)} ({data.index[0].date()} — {data.index[-1].date()})")

bt = Backtest(data, BasisAdxStrategy, cash=100_000, commission=0.001, finalize_trades=True)
stats = bt.run()

print(f"\n{'─' * 50}")
print(f"  Return      : {stats['Return [%]']:+6.2f}%")
print(f"  Equity      : ${stats['Equity Final [$]']:,.2f}")
print(f"  Buy&Hold    : {stats['Buy & Hold Return [%]']:.2f}%")
print(f"  Sharpe      : {stats['Sharpe Ratio']:.2f}")
print(f"  Max DD      : {stats['Max. Drawdown [%]']:.2f}%")
print(f"  # Trades    : {stats['# Trades']}")
print(f"  Win Rate    : {stats.get('Win Rate [%]', 0):.1f}%")
print(f"  Best/Worst  : {stats.get('Best Trade [%]', 0):.2f}% / "
      f"{stats.get('Worst Trade [%]', 0):.2f}%")

trades = stats['_trades']
if trades is not None and len(trades) > 0:
    print(f"\n  TRADES:")
    for _, t in trades.iterrows():
        et = t['EntryTime']; xt = t['ExitTime']
        ret = t.get('ReturnPct', 0) * 100
        print(f"    {et.date()} → {xt.date()}  "
              f"PnL: {t['PnL']:+8.2f}  Ret: {ret:+5.2f}%")

print(f"\n{'=' * 60}")
print("  ✅ Done")
print(f"{'=' * 60}")

out = Path(r'C:\Users\Acer\code\backtester\reports')
out.mkdir(exist_ok=True)
bt.plot(filename=str(out / 'TSLA_basis_adx.html'), open_browser=False)
print(f"  HTML: {out / 'TSLA_basis_adx.html'}")
