"""
Backtest Mean Reversion ADX — GBP/USD Daily, last 2 years.
"""
import numpy as np, pandas as pd, yfinance as yf, warnings
warnings.filterwarnings("ignore")
from backtesting import Backtest
from strategies.mean_rev_adx_strategy import MeanRevAdxStrategy

print("=" * 60)
print("  Mean Reversion ADX — GBP/USD Daily (2 tahun)")
print("=" * 60)

ticker = "GBPUSD=X"
data = yf.download(ticker, period="2y", interval="1d", auto_adjust=True)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data = data.dropna(subset=["Open","High","Low","Close"])
print(f"  Bars: {len(data)} ({data.index[0].date()} — {data.index[-1].date()})")

for adx_val in [20, 25, 30]:
    class Override(MeanRevAdxStrategy):
        adx_threshold = adx_val

    bt = Backtest(data, Override, cash=100_000, commission=0.0001, finalize_trades=True)
    stats = bt.run()
    wr = stats.get("Win Rate [%]", 0)
    print(f"  ADX < {adx_val:2d}: "
          f"Return {stats['Return [%]']:+6.2f}% | "
          f"Trades {stats['# Trades']:2d} | "
          f"Win% {wr:5.1f}% | "
          f"Sharpe {stats['Sharpe Ratio']:.2f} | "
          f"DD {stats['Max. Drawdown [%]']:.2f}%")

best = 20
print(f"\n\n{'=' * 60}")
print(f"  DETAIL: ADX < {best}")
print(f"{'=' * 60}")

class Best(MeanRevAdxStrategy):
    adx_threshold = best

bt = Backtest(data, Best, cash=100_000, commission=0.0001, finalize_trades=True)
stats = bt.run()

print(f"  Return      : {stats['Return [%]']:+6.2f}%")
print(f"  Equity      : ${stats['Equity Final [$]']:,.2f}")
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
