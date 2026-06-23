"""Compare BASELINE vs ADX valley on EURUSD 1D."""
import sys
sys.path.insert(0, r'C:\Users\satri\code\backtester')

from backtesting import Backtest
import pandas as pd

from strategies.basis_adx_strategy import BasisAdxStrategy
from strategies.basis_adx_simple_strategy import BasisAdxSimple

data = pd.read_csv(
    r'C:\Users\satri\code\backtester\EURUSD_1d_yf.txt',
    parse_dates=['Date'],
    index_col='Date',
)
data.columns = [c.strip() for c in data.columns]
print(f"Data: {len(data)} bars, {data.index[0].date()} → {data.index[-1].date()}")

results = []

for label, strat in [
    ("BASELINE: ADX>20 + ADX rising", BasisAdxSimple),
    ("VALLEY: ADX>20 + ADX valley", BasisAdxStrategy),
]:
    bt = Backtest(data, strat, cash=100_000, commission=0.001)
    stats = bt.run()
    trades = stats['_trades']
    n_trades = len(trades) if isinstance(trades, pd.DataFrame) else 0
    
    results.append({
        'Strategy': label,
        'Return %': round(stats['Return [%]'], 2),
        'CAGR %': round(stats.get('CAGR [%]', 0), 2),
        'Sharpe': round(stats['Sharpe Ratio'], 3),
        'Max DD %': round(stats['Max. Drawdown [%]'], 2),
        '# Trades': n_trades,
        'Win Rate %': round(stats['Win Rate [%]'], 1),
        'Profit Factor': round(stats['Profit Factor'], 3),
        'Avg Trade %': round(stats['Avg. Trade [%]'], 2),
        'Best Trade %': round(stats['Best Trade [%]'], 2),
        'Worst Trade %': round(stats['Worst Trade [%]'], 2),
    })

# Table
df = pd.DataFrame(results)
compare = df.set_index('Strategy').T

print("\n" + "=" * 110)
print("  EURUSD 1D — BASELINE vs ADX VALLEY  (2006–2026)")
print("=" * 110)

for col in compare.columns:
    print(f"\n  {'─'*50}")
    print(f"  ║  {col}")
    print(f"  {'─'*50}")
    for idx, val in compare[col].items():
        print(f"  ║  {idx:20s} → {val}")

print("\n\n  RAW TABLE:")
print(df.to_string(index=False))
print("\n  Done!")
