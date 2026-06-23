"""Compare BASELINE vs PRICE MOMENTUM on GC=F and EURUSD 1D."""
import sys
sys.path.insert(0, r'C:\Users\satri\code\backtester')

from backtesting import Backtest
import pandas as pd

from strategies.basis_adx_strategy import BasisAdxStrategy  # momentum
from strategies.basis_adx_simple_strategy import BasisAdxSimple  # baseline

pairs = [
    ("GC=F", r'C:\Users\satri\code\backtester\GC_F_1d.txt'),
    ("EURUSD", r'C:\Users\satri\code\backtester\EURUSD_1d_yf.txt'),
]

for pair_name, path in pairs:
    data = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    data.columns = [c.strip() for c in data.columns]
    print(f"\n{'='*90}")
    print(f"  {pair_name} 1D — {len(data)} bars")
    print(f"{'='*90}")

    results = []
    for label, strat in [
        ("BASELINE: ADX rising", BasisAdxSimple),
        ("MOMENTUM: C > C[10]", BasisAdxStrategy),
    ]:
        bt = Backtest(data, strat, cash=100_000, commission=0.001)
        stats = bt.run()
        trades = stats['_trades']
        n_trades = len(trades) if isinstance(trades, pd.DataFrame) else 0
        results.append({
            'Strategy': label,
            'Return %': round(stats['Return [%]'], 2),
            'Sharpe': round(stats['Sharpe Ratio'], 3),
            'Max DD %': round(stats['Max. Drawdown [%]'], 2),
            '# Trades': n_trades,
            'Win Rate %': round(stats['Win Rate [%]'], 1),
            'Profit Factor': round(stats['Profit Factor'], 3),
            'Avg Trade %': round(stats['Avg. Trade [%]'], 2),
        })

    df = pd.DataFrame(results)
    compare = df.set_index('Strategy').T
    for col in compare.columns:
        print(f"\n  {'─'*45}")
        print(f"  ║  {col}")
        print(f"  {'─'*45}")
        for idx, val in compare[col].items():
            print(f"  ║  {idx:20s} → {val}")
    print()
