"""
US Portfolio Backtest — FILTERED by ADX+SMA20 ranking.
Phase 1: Scan all stocks, rank by % bullish trend bars
Phase 2: Pick top N, run portfolio backtest with shared capital
"""
import sys, os, csv
sys.path.insert(0, r'C:\Users\satri\code\backtester')
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from portfolio.portfolio_backtest import (
    run_single_backtest, build_portfolio_equity, 
    print_portfolio_summary, print_per_ticker_summary, plot_portfolio
)

CAPITAL = 100000       # USD
TOP_N = 15             # Top N stocks to pick
PERIOD_BACKTEST = "5y"  # Backtest period
SCAN_PERIOD = "1y"     # Period untuk scanning trend

# Load tickers
tickers = []
csv_path = os.path.join(os.path.dirname(__file__), 'uslist.csv')
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        sym = row['Symbol'].strip()
        if not sym.startswith('^'):
            tickers.append(sym)

print("=" * 70)
print("  US PORTFOLIO — PHASE 1: RANKING STOCKS BY TREND STRENGTH")
print("=" * 70)
print(f"  Scanning {len(tickers)} stocks...")
print()

def calc_adx_sma_pct(ticker):
    """Calculate % of bars where ADX>25 + Close>SMA20 over 1y."""
    try:
        df = yf.download(ticker, period=SCAN_PERIOD, progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna(subset=['close', 'high', 'low'])
        if len(df) < 50:
            return None
        
        # SMA20
        df['sma20'] = df['close'].rolling(20).mean()
        
        # ADX
        high, low, close = df['high'].values, df['low'].values, df['close'].values
        tr = np.maximum(high - low, 
             np.maximum(np.abs(high - np.roll(close, 1)), 
                        np.abs(low - np.roll(close, 1))))
        up_move = np.diff(high, prepend=high[0])
        down_move = np.diff(low, prepend=low[0])
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        period = 14
        atr = pd.Series(tr).rolling(period).mean().values
        sp = pd.Series(plus_dm).rolling(period).mean().values
        sm = pd.Series(minus_dm).rolling(period).mean().values
        
        pdi = 100 * sp / np.where(atr > 0, atr, np.nan)
        mdi = 100 * sm / np.where(atr > 0, atr, np.nan)
        dx = 100 * np.abs(pdi - mdi) / np.where((pdi + mdi) > 0, pdi + mdi, np.nan)
        adx = pd.Series(dx).rolling(period).mean().values
        
        # Count bars
        total = 0
        bull = 0  # ADX>25 + Close>SMA20
        bear = 0
        for i in range(max(20, period), len(df)):
            if np.isnan(adx[i]) or np.isnan(df['sma20'].iloc[i]):
                continue
            total += 1
            c = float(df['close'].iloc[i])
            s = float(df['sma20'].iloc[i])
            a = float(adx[i])
            if a > 25:
                if c > s:
                    bull += 1
                else:
                    bear += 1
        
        if total == 0:
            return None
        
        bull_pct = bull / total * 100
        bear_pct = bear / total * 100
        
        # Current snapshot
        last_close = float(df['close'].iloc[-1])
        last_sma20 = float(df['sma20'].iloc[-1])
        last_adx = float(adx[-1])
        last_pdi = float(pdi[-1])
        last_mdi = float(mdi[-1])
        
        # Current condition check
        above_sma = last_close > last_sma20
        pdi_above = last_pdi > last_mdi
        
        return {
            'ticker': ticker,
            'bull_pct': round(bull_pct, 1),
            'bear_pct': round(bear_pct, 1),
            'side_pct': round(100 - bull_pct - bear_pct, 1),
            'price': round(last_close, 2),
            'adx': round(last_adx, 1),
            'pdi': round(last_pdi, 1),
            'mdi': round(last_mdi, 1),
            'above_sma': above_sma,
            'pdi_above': pdi_above,
        }
    except Exception:
        return None

rankings = []
for i, ticker in enumerate(tickers):
    result = calc_adx_sma_pct(ticker)
    if result is not None:
        rankings.append(result)
    if (i + 1) % 50 == 0:
        print(f"  Scanned {i+1}/{len(tickers)}... ({len(rankings)} valid)")

# Rank by bull_pct descending
rankings.sort(key=lambda x: x['bull_pct'], reverse=True)

# Top N
top_picks = rankings[:TOP_N]

print()
print("=" * 70)
print(f"  TOP {TOP_N} STOCKS — Ranked by ADX+SMA20 Bullish %")
print("=" * 70)
print(f"{'Rank':<5} {'Ticker':<10} {'Bull%':>7} {'Bear%':>7} {'Price':>10} {'ADX':>6} {'+DI/-DI':>12} {'Status':>10}")
print("-" * 70)
for i, r in enumerate(top_picks, 1):
    di = f"{r['pdi']}/{r['mdi']}"
    st = "✅" if r['above_sma'] and r['pdi_above'] else "⚠️"
    print(f"  #{i:<2}  {r['ticker']:<10} {r['bull_pct']:>6.1f}% {r['bear_pct']:>6.1f}% ${r['price']:>8,.2f} {r['adx']:>5.1f} {di:>11} {st}")

print()
print(f"  Other stocks with scores: {len(rankings)} total")
print(f"  Top pick bull% range: {top_picks[0]['bull_pct']:.1f}% — {top_picks[-1]['bull_pct']:.1f}%")

# ── Phase 2: Portfolio Backtest ──
print()
print("=" * 70)
print("  PHASE 2: PORTFOLIO BACKTEST (shared capital)")
print("=" * 70)

selected = [r['ticker'] for r in top_picks]
print(f"  Backtesting top {len(selected)} stocks over {PERIOD_BACKTEST}")
print()

all_trades = []
per_ticker_results = {}

for ticker in selected:
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

print()
portfolio = build_portfolio_equity(all_trades, CAPITAL)

if portfolio:
    print_portfolio_summary(portfolio, CAPITAL)
    
    active = {k: v for k, v in per_ticker_results.items() if v['n_trades'] > 0}
    print_per_ticker_summary(active)
    
    output_dir = os.path.join(r'C:\Users\satri\code\backtester', 'portfolio')
    chart_path = plot_portfolio(portfolio, output_dir)
    
    import json
    from datetime import datetime
    report = {
        'timestamp': datetime.now().isoformat(),
        'market': 'US (filtered)',
        'capital': CAPITAL,
        'period': PERIOD_BACKTEST,
        'ranking_period': SCAN_PERIOD,
        'top_n': TOP_N,
        'stocks_scanned': len(rankings),
        'selected_tickers': selected,
        'portfolio': {
            'total_return': portfolio['total_return'],
            'cagr': portfolio['cagr'],
            'max_dd': portfolio['max_dd'],
            'sharpe': portfolio['sharpe'],
            'profit_factor': portfolio['profit_factor'],
            'total_trades': portfolio['n_trades'],
            'win_rate': portfolio['win_rate'],
            'max_concurrent': portfolio['max_concurrent'],
            'skipped': portfolio.get('skipped', 0),
            'final_equity': portfolio['equity'].iloc[-1],
        },
    }
    with open(os.path.join(output_dir, 'portfolio_us_filtered_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # CSV
    eq_df = pd.DataFrame({
        'Date': portfolio['equity'].index,
        'Equity': portfolio['equity'].values,
        'Cash': portfolio['cash'].values,
        'Invested': portfolio['invested'].values,
        'OpenPositions': portfolio['open_positions'].values,
    })
    eq_df.to_csv(os.path.join(output_dir, 'portfolio_us_filtered_equity.csv'), index=False)
    
    if portfolio['trades'] is not None and len(portfolio['trades']) > 0:
        portfolio['trades'].to_csv(os.path.join(output_dir, 'portfolio_us_filtered_trades.csv'), index=False)
    
    print(f"\n  📁 Report saved")
    if chart_path:
        print(f"  🖼️  MEDIA:{chart_path}")
else:
    print("❌ No trades generated.")
