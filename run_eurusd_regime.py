"""
Backtest Regime-Switching Strategy — EUR/USD Daily, last 2 years.

Regime detection via trend_score over last 100 bars:
  score >= MIN_TREND_SCORE  →  Trend mode (Basis+ADX)
  score <  MIN_TREND_SCORE  →  Mean Reversion mode (RSI+BB)
"""
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from backtesting import Backtest
from strategies.regime_switching_strategy import RegimeSwitchingStrategy

print("=" * 60)
print("  REGIME-SWITCHING STRATEGY — EUR/USD Daily")
print("  TrendScore >= 30 → Basis+ADX  |  < 30 → Mean Reversion")
print("=" * 60)

ticker = "EURUSD=X"
print(f"\nDownloading {ticker} daily...")
data = yf.download(ticker, period="2y", interval="1d", auto_adjust=True)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data = data.dropna(subset=["Open", "High", "Low", "Close"])
print(f"  Bars: {len(data)} ({data.index[0].date()} — {data.index[-1].date()})")

# Test different min_trend_score thresholds
for threshold in [20, 25, 30, 35]:
    print(f"\n{'─' * 50}")
    print(f"  MIN_TREND_SCORE = {threshold}")
    print(f"{'─' * 50}")

    class Override(RegimeSwitchingStrategy):
        min_trend_score = threshold

    bt = Backtest(data, Override, cash=100_000, commission=0.0001, finalize_trades=True)
    stats = bt.run()

    print(f"  Return: {stats['Return [%]']:+.2f}%  |  "
          f"Sharpe: {stats['Sharpe Ratio']:.2f}  |  "
          f"DD: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"  Trades: {stats['# Trades']}  |  "
          f"Win%: {stats.get('Win Rate [%]', 0):.1f}%  |  "
          f"Best: {stats.get('Best Trade [%]', 0):.2f}%  |  "
          f"Worst: {stats.get('Worst Trade [%]', 0):.2f}%")

# Best config — detail
best = 25
print(f"\n\n{'=' * 60}")
print(f"  DETAIL: MIN_TREND_SCORE = {best}")
print(f"{'=' * 60}")

class Best(RegimeSwitchingStrategy):
    min_trend_score = best

bt = Backtest(data, Best, cash=100_000, commission=0.0001, finalize_trades=True)
stats = bt.run()

print(f"  Return        : {stats['Return [%]']:+6.2f}%")
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
              f"| Return: {trade.get('ReturnPct', 0)*100:+5.2f}%")

print(f"\n{'=' * 60}")
print("  ✅ Done")
print(f"{'=' * 60}")
