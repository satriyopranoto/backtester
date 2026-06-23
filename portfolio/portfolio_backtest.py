"""
Portfolio Backtester — Multiple stocks, single strategy, aggregate results.
    
Runs Basis ADX strategy on each ticker independently, then combines 
all trades into a portfolio timeline with aggregate equity curve, chart, CAGR.
"""

import sys
import os
import csv
import json
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Add project root for strategy import
PROJECT = r'C:\Users\satri\code\backtester'
sys.path.insert(0, PROJECT)

from backtesting import Backtest
from strategies.basis_adx_strategy import BasisAdxStrategy


# ── CONFIG ──────────────────────────────────────────────
CAPITAL = 100_000_000       # Rp (ID) or $6,000 (US)
COMMISSION = 0.001          # 0.1%
PERIOD = "2y"               # Data period

# ── Portfolio Tickers ────────────────────────────────────
SCREENER_CSV = os.path.join(os.path.dirname(__file__), 'screener_cache.csv')

def load_tickers_from_screener(csv_path=SCREENER_CSV):
    try:
        with open(csv_path) as f:
            next(f)
            reader = csv.DictReader(f)
            buys = [r['ticker'].strip() for r in reader if r['recommendation'].strip() == 'BUY']
            return buys
    except Exception:
        return []

tickers_from_screener = load_tickers_from_screener()
if tickers_from_screener:
    PORTFOLIO = tickers_from_screener
    print(f"📋 Loaded {len(PORTFOLIO)} BUY signals from screener")
else:
    PORTFOLIO = [
        "BSSR.JK", "DATA.JK", "ADES.JK", "MORA.JK", "GGRM.JK",
        "JARR.JK", "CASS.JK",
    ]
    print("📋 Using manual ticker list")


def get_currency(ticker):
    return "Rp" if ticker.upper().endswith(".JK") else "$"


def run_single_backtest(ticker, cash=CAPITAL):
    """Run Basis ADX strategy on a single ticker. Returns (stats, trades_df)."""
    print(f"  ⏳ {ticker}...", end=" ", flush=True)
    try:
        import yfinance as yf
        df = yf.download(ticker, period=PERIOD, progress=False, auto_adjust=True)
        if df.empty:
            print("❌ No data")
            return None, None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        for c in ['open', 'high', 'low', 'close', 'volume']:
            if c not in df.columns:
                print(f"❌ Missing col {c}")
                return None, None
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        if len(df) < 50:
            print(f"❌ Too few bars ({len(df)})")
            return None, None
        df_bt = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        })
        bt = Backtest(df_bt, BasisAdxStrategy, cash=cash, commission=COMMISSION)
        stats = bt.run()
        n_trades = int(stats['# Trades']) if '# Trades' in stats else 0
        ret = stats['Return [%]'] if 'Return [%]' in stats else 0
        trades_df = stats.get('_trades', None)
        if trades_df is not None and isinstance(trades_df, pd.DataFrame) and len(trades_df) > 0:
            trades_df['Ticker'] = ticker
            print(f"✅ {n_trades} trades, {ret:+.2f}%")
            return stats, trades_df
        else:
            print(f"⏭️  0 trades")
            return stats, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None


def build_portfolio_equity(all_trades, cash=CAPITAL):
    """
    Build aggregate portfolio equity + cash curves from all trades
    dengan SHARED CAPITAL SIMULATION — cash terbatas, tidak bisa
    entry posisi baru kalau duit habis.
    """
    if not all_trades:
        return None
    
    combined = pd.concat(all_trades, ignore_index=True)
    
    # Collect all dates
    all_dates = set()
    for t in all_trades:
        for col in ['EntryTime', 'ExitTime']:
            for date in t[col]:
                all_dates.add(pd.Timestamp(date).normalize())
    if not all_dates:
        return None
    
    start_date = min(all_dates) - timedelta(days=1)
    end_date = max(all_dates) + timedelta(days=1)
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    # ── Process trades chronologically with shared capital ──
    trade_events = []
    for _, trade in combined.iterrows():
        entry_d = pd.Timestamp(trade['EntryTime']).normalize()
        exit_d = pd.Timestamp(trade['ExitTime']).normalize()
        size = trade['Size']
        entry_price = trade['EntryPrice']
        exit_price = trade['ExitPrice']
        cost = size * entry_price
        exit_value = size * exit_price
        
        trade_events.append({
            'date': entry_d, 'type': 'entry', 'cost': cost,
            'exit_value': exit_value, 'ticker': trade.get('Ticker', '?'),
        })
        trade_events.append({
            'date': exit_d, 'type': 'exit', 'cost': -exit_value,
            'exit_value': 0, 'ticker': trade.get('Ticker', '?'),
        })
    
    trade_events.sort(key=lambda x: (x['date'], x['type']))
    
    # Simulate FIFO portfolio
    current_cash = float(cash)
    active_positions = []
    skipped_trades = 0
    
    cash_at_date = {}
    positions_at_date = {}
    invested_at_date = {}
    
    for d in date_range:
        day_events = [ev for ev in trade_events if ev['date'] == d]
        
        for ev in day_events:
            if ev['type'] == 'entry':
                if current_cash >= ev['cost']:
                    current_cash -= ev['cost']
                    active_positions.append({
                        'cost': ev['cost'], 'exit_value': ev['exit_value'],
                    })
                else:
                    skipped_trades += 1
            elif ev['type'] == 'exit':
                if active_positions:
                    pos = active_positions.pop(0)
                    current_cash += pos['exit_value']
        
        cash_at_date[d] = current_cash
        invested_at_date[d] = sum(p['cost'] for p in active_positions)
        positions_at_date[d] = len(active_positions)
    
    equity = pd.Series({d: cash_at_date[d] + invested_at_date[d] for d in date_range}).sort_index()
    cash_series = pd.Series(cash_at_date).sort_index()
    invested_series = pd.Series(invested_at_date).sort_index()
    open_pos_series = pd.Series(positions_at_date).sort_index()
    
    # Stats
    total_return = (equity.iloc[-1] / cash - 1) * 100
    total_pnl = equity.iloc[-1] - cash
    
    # Drawdown
    peak = equity.expanding().max()
    dd = (equity - peak) / peak * 100
    max_dd = dd.min()
    
    # CAGR
    days = (equity.index[-1] - equity.index[0]).days
    if days > 0:
        cagr = ((equity.iloc[-1] / cash) ** (365.0 / days) - 1) * 100
    else:
        cagr = 0.0
    
    # Win rate
    n_total = len(combined)
    n_win = (combined['PnL'] > 0).sum()
    win_rate = n_win / n_total * 100 if n_total > 0 else 0
    
    # Sharpe
    daily_returns = equity.pct_change().dropna()
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
    
    # Max concurrent
    max_concurrent = int(open_pos_series.max())
    
    # Profit factor
    gross_profit = combined[combined['PnL'] > 0]['PnL'].sum()
    gross_loss = abs(combined[combined['PnL'] < 0]['PnL'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return {
        'equity': equity,
        'cash': cash_series,
        'invested': invested_series,
        'open_positions': open_pos_series,
        'total_pnl': total_pnl,
        'total_return': total_return,
        'cagr': cagr,
        'max_dd': max_dd,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'n_trades': n_total,
        'n_wins': int(n_win),
        'max_concurrent': max_concurrent,
        'sharpe': sharpe,
        'skipped': skipped_trades,
        'trades': combined,
        'start_date': equity.index[0],
        'end_date': equity.index[-1],
    }


def plot_portfolio(result, output_path):
    """Generate equity curve chart using matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("  ⚠️  matplotlib not installed — skipping chart")
        return None
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9), 
                                         gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.patch.set_facecolor('#1a1612')
    
    dates = result['equity'].index
    equity = result['equity'].values
    cash_s = result['cash'].values
    invested = result['invested'].values
    
    # ── Panel 1: Equity + Cash ──
    ax1.fill_between(dates, equity, cash_s, alpha=0.15, color='#d4af37')
    ax1.plot(dates, equity, color='#d4af37', linewidth=1.5, label='Total Equity')
    ax1.plot(dates, cash_s, color='#4ade80', linewidth=0.8, alpha=0.7, label='Cash')
    ax1.plot(dates, invested, color='#60a5fa', linewidth=0.8, alpha=0.7, label='Invested')
    ax1.axhline(y=equity[0], color='#666', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Style
    for spine in ax1.spines.values():
        spine.set_color('#333')
    ax1.tick_params(colors='#99907c')
    ax1.set_ylabel('Equity (Rp)', color='#99907c')
    ax1.legend(loc='upper left', facecolor='#1a1612', edgecolor='#333', 
               labelcolor='#eae1d4', fontsize=9)
    ax1.set_facecolor('#1a1612')
    ax1.grid(True, alpha=0.08, color='#d4af37')
    ax1.set_title('Portfolio Equity Curve', color='#d4af37', fontsize=14, fontweight='bold')
    
    # ── Panel 2: Drawdown ──
    peak = pd.Series(equity).expanding().max().values
    dd = (equity - peak) / peak * 100
    ax2.fill_between(dates, 0, dd, color='#f87171', alpha=0.4)
    ax2.plot(dates, dd, color='#f87171', linewidth=0.8)
    for spine in ax2.spines.values():
        spine.set_color('#333')
    ax2.tick_params(colors='#99907c')
    ax2.set_ylabel('Drawdown %', color='#99907c')
    ax2.set_facecolor('#1a1612')
    ax2.grid(True, alpha=0.08, color='#d4af37')
    
    # ── Panel 3: Open Positions ──
    ax3.fill_between(dates, 0, result['open_positions'].values, 
                     color='#60a5fa', alpha=0.3, step='mid')
    ax3.plot(dates, result['open_positions'].values, 
             color='#60a5fa', linewidth=0.8, drawstyle='steps-mid')
    for spine in ax3.spines.values():
        spine.set_color('#333')
    ax3.tick_params(colors='#99907c')
    ax3.set_ylabel('Open Positions', color='#99907c')
    ax3.set_xlabel('Date', color='#99907c')
    ax3.set_facecolor('#1a1612')
    ax3.grid(True, alpha=0.08, color='#d4af37')
    ax3.set_ylim(bottom=0)
    
    # Date formatting
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    plt.tight_layout()
    chart_path = os.path.join(output_path, 'portfolio_chart.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#1a1612')
    plt.close()
    print(f"  🖼️  Chart saved: {chart_path}")
    return chart_path


def print_portfolio_summary(result, cash):
    """Print portfolio-level statistics with CAGR."""
    if result is None:
        print("❌ No portfolio data")
        return
    
    curr = "Rp"
    days = (result['end_date'] - result['start_date']).days
    
    print()
    print("=" * 65)
    print("  PORTFOLIO SUMMARY")
    print("=" * 65)
    print(f"  Initial Capital  : {curr} {cash:,.0f}")
    print(f"  Final Equity     : {curr} {result['equity'].iloc[-1]:,.0f}")
    print(f"  Total PnL        : {curr} {result['total_pnl']:,.0f}")
    print(f"  Total Return     : {result['total_return']:+.2f}%")
    print(f"  CAGR             : {result['cagr']:+.2f}%")
    print(f"  Max Drawdown     : {result['max_dd']:.2f}%")
    print(f"  Profit Factor    : {result['profit_factor']:.2f}")
    print(f"  Sharpe Ratio     : {result['sharpe']:.2f}")
    print(f"  Total Trades     : {result['n_trades']}")
    print(f"  Win Rate         : {result['n_wins']}/{result['n_trades']} ({result['win_rate']:.1f}%)")
    print(f"  Max Concurrent   : {result['max_concurrent']} positions")
    if result.get('skipped', 0) > 0:
        print(f"  Skipped (no cash) : {result['skipped']} trades")
    print(f"  Period           : {result['start_date'].date()} → {result['end_date'].date()} ({days} days)")
    print("=" * 65)


def print_per_ticker_summary(results):
    """Print per-ticker breakdown."""
    print()
    header = f"{'Ticker':<12} {'Trades':>7} {'Win%':>7} {'Return%':>10} {'Avg PnL':>13} {'MaxDD%':>8}"
    print(header)
    print("-" * len(header))
    sorted_tickers = sorted(results.items(), key=lambda x: x[1]['return'], reverse=True)
    for ticker, r in sorted_tickers:
        avg_pnl = r['trades']['PnL'].mean() if r['trades'] is not None and len(r['trades']) > 0 else 0
        wr_display = f"{r['win_rate']:.1f}%" if r['n_trades'] > 0 else "-"
        print(f"{ticker:<12} {r['n_trades']:>7} {wr_display:>7} {r['return']:>+9.2f}% {avg_pnl:>+11,.0f} {r['max_dd']:>7.2f}%")


# ── Main ────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("  PORTFOLIO BACKTESTER — Basis ADX Strategy")
    print("=" * 65)
    print(f"  Capital: Rp {CAPITAL:,}")
    print(f"  Period:  {PERIOD}")
    print(f"  Tickers: {len(PORTFOLIO)}")
    print()
    
    all_trades = []
    per_ticker_results = {}
    
    for ticker in PORTFOLIO:
        stats, trades = run_single_backtest(ticker, CAPITAL)
        if stats is not None:
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
    
    # ── Portfolio-level ──
    print()
    print("─" * 65)
    print("  BUILDING PORTFOLIO EQUITY CURVE...")
    
    portfolio = build_portfolio_equity(all_trades, CAPITAL)
    output_dir = os.path.join(PROJECT, 'portfolio')
    os.makedirs(output_dir, exist_ok=True)
    
    if portfolio:
        print_portfolio_summary(portfolio, CAPITAL)
        print_per_ticker_summary(per_ticker_results)
        
        # Save chart
        chart_path = plot_portfolio(portfolio, output_dir)
        
        # Save reports
        report = {
            'timestamp': datetime.now().isoformat(),
            'capital': CAPITAL,
            'period': PERIOD,
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
            'tickers': {t: {k: v for k, v in r.items() if k != 'trades'}
                       for t, r in per_ticker_results.items()}
        }
        with open(os.path.join(output_dir, 'portfolio_report.json'), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save equity curve CSV
        eq_df = pd.DataFrame({
            'Date': portfolio['equity'].index,
            'Equity': portfolio['equity'].values,
            'Cash': portfolio['cash'].values,
            'Invested': portfolio['invested'].values,
            'OpenPositions': portfolio['open_positions'].values,
            'Drawdown%': ((portfolio['equity'] - portfolio['equity'].expanding().max()) / portfolio['equity'].expanding().max() * 100).values,
        })
        eq_df.to_csv(os.path.join(output_dir, 'portfolio_equity.csv'), index=False)
        
        if portfolio['trades'] is not None and len(portfolio['trades']) > 0:
            portfolio['trades'].to_csv(os.path.join(output_dir, 'portfolio_trades.csv'), index=False)
        
        print(f"\n  📁 Report saved to: {output_dir}/")
        if chart_path:
            print(f"  🖼️  Include MEDIA:{chart_path} to see the chart")
    else:
        print("❌ No trades generated across all tickers.")
