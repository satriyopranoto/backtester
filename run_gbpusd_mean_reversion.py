"""
Backtest Mean Reversion — GBP/USD Daily, last 2 years.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from backtesting import Backtest
from strategies.mean_reversion_strategy import MeanReversionStrategy

ticker = "GBPUSD=X"
print("=" * 60)
print("  MEAN REVERSION — GBP/USD Daily")
print("  (RSI+BB: entry oversold, exit overbought)")
print("=" * 60)

data = yf.download(ticker, period="2y", interval="1d", auto_adjust=True)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data = data.dropna(subset=["Open", "High", "Low", "Close"])
print(f"\n  Bars: {len(data)} ({data.index[0].date()} — {data.index[-1].date()})")
r = f"{data['Low'].min():.4f} — {data['High'].max():.4f}"
print(f"  Range: {r}")

for rsi_entry in [20, 25, 30]:
    for rsi_exit in [70, 65]:
        class Override(MeanReversionStrategy):
            rsi_oversold = rsi_entry
            rsi_overbought = rsi_exit

        bt = Backtest(data, Override, cash=100_000, commission=0.0001, finalize_trades=True)
        stats = bt.run()
        wr = stats.get("Win Rate [%]", 0)
        print(f"  RSI {rsi_entry:2d}→{rsi_exit:2d}: "
              f"Return {stats['Return [%]']:+6.2f}% | "
              f"Trades {stats['# Trades']:2d} | "
              f"Win% {wr:5.1f}% | "
              f"Sharpe {stats['Sharpe Ratio']:.2f} | "
              f"DD {stats['Max. Drawdown [%]']:.2f}%")

# Best config detail
best_e, best_x = 30, 70
print(f"\n\n{'=' * 60}")
print(f"  DETAIL: RSI {best_e}→{best_x}")
print(f"{'=' * 60}")

class Best(MeanReversionStrategy):
    rsi_oversold = best_e
    rsi_overbought = best_x

bt = Backtest(data, Best, cash=100_000, commission=0.0001, finalize_trades=True)
stats = bt.run()
eq_final = stats['Equity Final [$]']
bt_return = stats['Return [%]']
sharpe = stats['Sharpe Ratio']
mdd = stats['Max. Drawdown [%]']
n_trades = stats['# Trades']
wr = stats.get('Win Rate [%]', 0)
bt_best = stats.get('Best Trade [%]', 0)
bt_worst = stats.get('Worst Trade [%]', 0)

print(f"  Return      : {bt_return:+6.2f}%")
print(f"  Equity      : ${eq_final:,.2f}")
print(f"  Sharpe      : {sharpe:.2f}")
print(f"  Max DD      : {mdd:.2f}%")
print(f"  # Trades    : {n_trades}")
print(f"  Win Rate    : {wr:.1f}%")
print(f"  Best/Worst  : {bt_best:.2f}% / {bt_worst:.2f}%")

trades = stats['_trades']
if trades is not None and len(trades) > 0:
    print(f"\n  TRADES:")
    for _, t in trades.iterrows():
        et = t['EntryTime']
        xt = t['ExitTime']
        ret = t.get('ReturnPct', 0) * 100
        print(f"    {et.date()} → {xt.date()}  "
              f"PnL: {t['PnL']:+8.2f}  "
              f"Ret: {ret:+5.2f}%")

print(f"\n{'=' * 60}")
print("  ✅ Done")
print(f"{'=' * 60}")
