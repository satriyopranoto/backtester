"""
Backtest Mean Reversion — EUR/USD Daily, last 2 years.

Entry: RSI < 20 AND Low < Lower Bollinger Band
Exit : RSI > 70 AND High > Upper Bollinger Band
         or SL / TP (same as other strategies)
"""
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from backtesting import Backtest
from strategies.mean_reversion_strategy import MeanReversionStrategy

print("=" * 60)
print("  MEAN REVERSION — EUR/USD Daily")
print("  RSI<20 & Low<LowerBB  →  RSI>70 & High>UpperBB")
print("=" * 60)

ticker = "EURUSD=X"
print(f"\nDownloading {ticker} daily...")
data = yf.download(ticker, period="2y", interval="1d", auto_adjust=True)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data = data.dropna(subset=["Open", "High", "Low", "Close"])
print(f"  Bars: {len(data)} ({data.index[0].date()} — {data.index[-1].date()})")
print(f"  Range: {data['Low'].min():.4f} — {data['High'].max():.4f}")

# Test with different RSI thresholds
for rsi_entry in [20, 25, 30]:
    for rsi_exit in [70, 65]:
        print(f"\n{'─' * 50}")
        print(f"  RSI entry={rsi_entry}, RSI exit={rsi_exit}")
        print(f"{'─' * 50}")

        class MeanRevOverride(MeanReversionStrategy):
            rsi_oversold = rsi_entry
            rsi_overbought = rsi_exit

        bt = Backtest(data, MeanRevOverride, cash=100_000, commission=0.0001, finalize_trades=True)
        stats = bt.run()

        print(f"  Return: {stats['Return [%]']:.2f}%  |  "
              f"Sharpe: {stats['Sharpe Ratio']:.2f}  |  "
              f"DD: {stats['Max. Drawdown [%]']:.2f}%")
        print(f"  Trades: {stats['# Trades']}  |  "
              f"Win%: {stats.get('Win Rate [%]', 0):.1f}%  |  "
              f"Best: {stats.get('Best Trade [%]', 0):.2f}%  |  "
              f"Worst: {stats.get('Worst Trade [%]', 0):.2f}%")

# Best config — detail
best_entry, best_exit = 25, 65
print(f"\n\n{'=' * 60}")
print(f"  DETAIL: RSI entry={best_entry}, exit={best_exit}")
print(f"{'=' * 60}")

class BestMeanRev(MeanReversionStrategy):
    rsi_oversold = best_entry
    rsi_overbought = best_exit

bt = Backtest(data, BestMeanRev, cash=100_000, commission=0.0001, finalize_trades=True)
stats = bt.run()

print(f"  Return        : {stats['Return [%]']:.2f}%")
print(f"  Equity Final  : ${stats['Equity Final [$]']:,.2f}")
print(f"  Sharpe Ratio  : {stats['Sharpe Ratio']:.2f}")
print(f"  Max DD        : {stats['Max. Drawdown [%]']:.2f}%")
print(f"  # Trades      : {stats['# Trades']}")
print(f"  Win Rate      : {stats.get('Win Rate [%]', 0):.1f}%")
print(f"  Best Trade    : {stats.get('Best Trade [%]', 0):.2f}%")
print(f"  Worst Trade   : {stats.get('Worst Trade [%]', 0):.2f}%")

# Show trades
trades = stats['_trades']
if trades is not None and len(trades) > 0:
    print(f"\n{'─' * 60}")
    print("  TRADES")
    print(f"{'─' * 60}")
    for i, trade in trades.iterrows():
        et = trade['EntryTime']
        xt = trade['ExitTime']
        print(f"  {et.date() if hasattr(et, 'date') else et}  →  "
              f"{xt.date() if hasattr(xt, 'date') else xt}  "
              f"| PnL: {trade['PnL']:+7.2f}  "
              f"| Return: {trade.get('ReturnPct', 0)*100:+.2f}%")

print(f"\n{'=' * 60}")
print("  ✅ Done")
print(f"{'=' * 60}")
