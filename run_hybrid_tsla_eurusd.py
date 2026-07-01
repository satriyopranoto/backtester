"""
Backtest Hybrid — Basis+ADX Entry + Mean Rev ADX Exit
"""
import numpy as np, pandas as pd, yfinance as yf, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from backtesting import Backtest
from strategies.hybrid_basis_adx_strategy import HybridBasisAdxStrategy

out = Path(r'C:\Users\Acer\code\backtester\reports')
out.mkdir(exist_ok=True)

for ticker, period, label in [("TSLA", "5y", "TSLA (5 tahun)"), ("EURUSD=X", "2y", "EUR/USD (2 tahun)")]:
    print(f"\n{'=' * 60}")
    print(f"  HYBRID — {label}")
    print(f"  (Entry: Basis+ADX || Exit: Mean Rev ADX)")
    print(f"{'=' * 60}")

    data = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.dropna(subset=["Open","High","Low","Close"])
    print(f"  Bars: {len(data)} ({data.index[0].date()} — {data.index[-1].date()})")

    comm = 0.001 if ticker == "TSLA" else 0.0001
    bt = Backtest(data, HybridBasisAdxStrategy, cash=100_000, commission=comm, finalize_trades=True)
    stats = bt.run()

    print(f"\n  Return      : {stats['Return [%]']:+6.2f}%")
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

    fname = f"{ticker.split('=')[0] if '=' in ticker else ticker}_hybrid.html"
    bt.plot(filename=str(out / fname), open_browser=False)
    print(f"\n  HTML: {out / fname}")

print(f"\n{'=' * 60}")
print("  ✅ Done")
print(f"{'=' * 60}")
