"""Run Basis ADX (valley) on GC=F daily."""
import sys
sys.path.insert(0, r'C:\Users\satri\code\backtester')

from backtesting import Backtest
from strategies.basis_adx_strategy import BasisAdxStrategy
import pandas as pd

data = pd.read_csv(
    r'C:\Users\satri\code\backtester\GC_F_1d.txt',
    parse_dates=['Date'],
    index_col='Date',
)
data.columns = [c.strip() for c in data.columns]

bt = Backtest(
    data,
    BasisAdxStrategy,
    cash=100_000,
    commission=0.001,
)

stats = bt.run()
print(stats)
print("\n=== TRADES ===")
trades = stats['_trades']
if isinstance(trades, pd.DataFrame) and len(trades) > 0:
    print(trades.to_string())
else:
    print("No trades")

# Save report
bt.plot(filename=r'C:\Users\satri\code\backtester\reports\GC_F_basis_adx_valley_1d.html', open_browser=False)
print("\nReport saved.")
