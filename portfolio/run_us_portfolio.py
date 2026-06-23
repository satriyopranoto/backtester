"""
Run portfolio backtester on US stocks from uslist.csv
Capital: $6,000 USD | Period: 5y | Commission: 0.001
"""
import sys, os, csv
sys.path.insert(0, r'C:\Users\satri\code\backtester')

from portfolio.portfolio_backtest import (
    run_single_backtest, build_portfolio_equity, 
    print_portfolio_summary, print_per_ticker_summary, plot_portfolio
)

CAPITAL = 100000     # USD
COMMISSION = 0.001
PERIOD = "5y"
MIN_TRADES = 2

# Load tickers from uslist.csv, exclude indices (^)
tickers = []
csv_path = os.path.join(os.path.dirname(__file__), 'uslist.csv')
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        sym = row['Symbol'].strip()
        if not sym.startswith('^'):
            tickers.append(sym)

print("=" * 65)
print("  PORTFOLIO BACKTESTER — US Market")
print("=" * 65)
print(f"  Capital: ${CAPITAL:,}")
print(f"  Period:  {PERIOD}")
print(f"  Tickers: {len(tickers)} (from uslist.csv)")
print()

all_trades = []
per_ticker_results = {}
total_stocks = 0

for ticker in tickers:
    stats, trades = run_single_backtest(ticker, CAPITAL)
    if stats is not None:
        total_stocks += 1
        n_trades = int(stats['# Trades']) if '# Trades' in stats else 0
        pnl_pct = stats['Return [%]'] if 'Return [%]' in stats else 0
        max_dd = stats['Max. Drawdown [%]'] if 'Max. Drawdown [%]' in stats else 0
        if trades is not None and len(trades) > 0:
            n_win = (trades['PnL'] > 0).sum()
            wr = n_win / len(trades) * 100
            all_trades.append(trades)
        else:
            n_win = 0
            wr = 0
        per_ticker_results[ticker] = {
            'n_trades': n_trades,
            'win_rate': wr,
            'return': pnl_pct,
            'max_dd': max_dd,
            'trades': trades,
        }

print()
print("─" * 65)
print(f"  TOTAL: {total_stocks} stocks analyzed, {len(per_ticker_results)} with trades")
print("  BUILDING PORTFOLIO EQUITY CURVE...")

portfolio = build_portfolio_equity(all_trades, CAPITAL)

if portfolio:
    print_portfolio_summary(portfolio, CAPITAL)
    
    # Filter to only stocks with trades
    active_results = {k: v for k, v in per_ticker_results.items() if v['n_trades'] > 0}
    print_per_ticker_summary(active_results)
    
    # Save chart
    output_dir = os.path.join(r'C:\Users\satri\code\backtester', 'portfolio')
    chart_path = plot_portfolio(portfolio, output_dir)
    
    import json
    from datetime import datetime
    report = {
        'timestamp': datetime.now().isoformat(),
        'market': 'US',
        'capital': CAPITAL,
        'period': PERIOD,
        'stocks_analyzed': total_stocks,
        'stocks_with_trades': len(active_results),
        'portfolio': {
            'total_return': portfolio['total_return'],
            'cagr': portfolio['cagr'],
            'max_dd': portfolio['max_dd'],
            'sharpe': portfolio['sharpe'],
            'profit_factor': portfolio['profit_factor'],
            'total_trades': portfolio['n_trades'],
            'win_rate': portfolio['win_rate'],
            'max_concurrent': portfolio['max_concurrent'],
            'final_equity': portfolio['equity'].iloc[-1],
        },
    }
    with open(os.path.join(output_dir, 'portfolio_us_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Save equity CSV
    import pandas as pd
    eq_df = pd.DataFrame({
        'Date': portfolio['equity'].index,
        'Equity': portfolio['equity'].values,
        'Cash': portfolio['cash'].values,
        'Invested': portfolio['invested'].values,
        'OpenPositions': portfolio['open_positions'].values,
    })
    eq_df.to_csv(os.path.join(output_dir, 'portfolio_us_equity.csv'), index=False)
    
    if portfolio['trades'] is not None and len(portfolio['trades']) > 0:
        portfolio['trades'].to_csv(os.path.join(output_dir, 'portfolio_us_trades.csv'), index=False)
    
    print(f"\n  📁 US portfolio report saved")
else:
    print("❌ No trades generated.")
