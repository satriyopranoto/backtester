#!/usr/bin/env python3
"""Backtester — CLI entry point.

Download data via yfinance, run backtest, tampilkan hasil.
Supports multiple strategies:
  - sma     : SMA Crossover (simple)
  - bb_adx  : BB Breakout + ADX Confirmation (from stocktrade)
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf
from backtesting import Backtest

from strategies.sma_crossover import SmaCrossover
from strategies.bb_adx_strategy import BbAdxStrategy
from strategies.high_breakout_adx import HighBreakoutAdx
from strategies.basis_adx_strategy import BasisAdxStrategy


# ── Data helpers ──────────────────────────────────────────────


def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data dari Yahoo Finance."""
    print(f"  📥 Download {ticker} dari {start} ke {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        print(f"  ❌ Data kosong untuk {ticker}. Cek ticker / tanggal.")
        sys.exit(1)

    # YFinance bisa return MultiIndex columns; flatten to single index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"  ✅ {len(df)} baris ({df.index[0].date()} — {df.index[-1].date()})")
    return df


# ── Stats printer ────────────────────────────────────────────


def get_currency(ticker: str, cash: float) -> str:
    """Detect currency from ticker suffix."""
    if ticker.upper().endswith(".JK"):
        return "Rp"
    return "$"


def print_stats(stats, initial_cash: float = 100_000_000, ticker: str = "BBCA.JK") -> None:
    """Pretty-print backtest stats (safe for 0-trade scenarios)."""
    eq_final = stats['Equity Final [$]']
    pnl = eq_final - initial_cash
    currency = get_currency(ticker, initial_cash)

    print(f"\n{'='*50}")
    print(f"  📊 HASIL BACKTEST")
    print(f"{'='*50}")
    print(f"  Start         : {stats['Start']}")
    print(f"  End           : {stats['End']}")
    print(f"  Return        : {stats['Return [%]']:.2f}%")
    print(f"  Buy & Hold    : {stats['Buy & Hold Return [%]']:.2f}%")
    print(f"  PnL Total     : {currency} {pnl:,.0f}")
    print(f"  Equity Final  : {currency} {eq_final:,.0f}")
    print(f"  Equity Peak   : {currency} {stats['Equity Peak [$]']:,.0f}")
    print(f"  Max Drawdown  : {stats['Max. Drawdown [%]']:.2f}%")
    print(f"  Sharpe Ratio  : {stats['Sharpe Ratio']:.2f}")
    print(f"  Trades        : {stats['# Trades']}")
    print(f"  Win Rate      : {stats.get('Win Rate [%]', float('nan')):.1f}%")
    print(f"  Best Trade    : {stats.get('Best Trade [%]', float('nan')):.2f}%")
    print(f"  Worst Trade   : {stats.get('Worst Trade [%]', float('nan')):.2f}%")
    print(f"  Avg Trade     : {stats.get('Avg Trade [%]', float('nan')):.2f}%")
    # Rincian trade terakhir
    trades = stats.get('_trades', None)
    if trades is not None and len(trades) > 0:
        print(f"  ──────────────────────────────────")
        for i, t in trades.tail(5).iterrows():
            ret = t.ReturnPct * 100
            if ret >= 0:
                emoji = "🔥"
                annot = ""
            else:
                emoji = "❌"
                annot = " cut loss"
            print(f"  Trade #{i}: {t.EntryTime.date()} → {t.ExitTime.date()}  "
                  f"| PnL: {ret:+.2f}%  ({currency} {t.PnL:+,.0f}) {emoji}{annot}")
    print(f"{'='*50}\n")


# ── Strategy router ──────────────────────────────────────────


def build_strategy_class(name: str, args) -> type:
    """Return a dynamically-created Strategy subclass with
    user-supplied parameter overrides."""
    if name == "sma":
        return type(
            "SmaCrossoverCustom",
            (SmaCrossover,),
            {
                "short_window": args.short_window,
                "long_window": args.long_window,
            },
        )
    elif name == "bb_adx":
        return type(
            "BbAdxStrategyCustom",
            (BbAdxStrategy,),
            {
                "bb_period": args.bb_period,
                "bb_std": args.bb_std,
                "adx_period": args.adx_period,
                "sl_multiple": args.sl_multiple,
                "sl_period": args.sl_period,
                "tp_min_profit_pct": args.tp_min_profit_pct,
                "risk_pct": args.risk_pct,
            },
        )
    elif name == "high_breakout":
        return type(
            "HighBreakoutAdxCustom",
            (HighBreakoutAdx,),
            {
                "hh_window": args.hh_window,
                "adx_period": args.adx_period,
                "sl_multiple": args.sl_multiple,
                "sl_period": args.sl_period,
                "tp_min_profit_pct": args.tp_min_profit_pct,
                "risk_pct": args.risk_pct,
            },
        )
    elif name == "basis_adx":
        return type(
            "BasisAdxStrategyCustom",
            (BasisAdxStrategy,),
            {
                "bb_period": args.bb_period,
                "adx_period": args.adx_period,
                "sl_multiple": args.sl_multiple,
                "sl_period": args.sl_period,
                "risk_pct": args.risk_pct,
            },
        )
    else:
        raise ValueError(f"Strategy tidak dikenal: {name}")


# ── CLI ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Backtester — Backtesting.py based trading strategy tester"
    )

    # Global
    parser.add_argument(
        "--strategy", choices=["sma", "bb_adx", "high_breakout", "basis_adx"], default="high_breakout",
        help="Strategy to use (default: bb_adx)",
    )
    parser.add_argument(
        "--ticker", default="BBCA.JK",
        help="Yahoo Finance ticker (default: BBCA.JK)",
    )
    parser.add_argument(
        "--start", default="2023-01-01",
        help="Start date (YYYY-MM-DD, default: 2023-01-01)",
    )
    parser.add_argument(
        "--end", default="2024-12-31",
        help="End date (YYYY-MM-DD, default: 2024-12-31)",
    )
    parser.add_argument(
        "--cash", type=float, default=100_000_000,
        help="Initial cash (default: 100,000,000 IDR / 6000 USD)",
    )
    parser.add_argument(
        "--risk-pct", type=float, default=1.0,
        help="Risk per trade as %% of equity (default: 1.0%%)",
    )
    parser.add_argument(
        "--commission", type=float, default=0.001,
        help="Commission per trade (default: 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Simpan laporan HTML ke folder reports/",
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="Jalankan optimasi parameter setelah backtest",
    )

    # Strategy-specific: SMA
    parser.add_argument(
        "--short-window", type=int, default=10,
        help="SMA short window (default: 10)",
    )
    parser.add_argument(
        "--long-window", type=int, default=30,
        help="SMA long window (default: 30)",
    )

    # Strategy-specific: BB + ADX
    parser.add_argument(
        "--bb-period", type=int, default=20,
        help="BB period (default: 20)",
    )
    parser.add_argument(
        "--bb-std", type=float, default=2.0,
        help="BB std dev multiplier (default: 2.0)",
    )

    # Strategy-specific: High Breakout + ADX
    parser.add_argument(
        "--hh-window", type=int, default=40,
        help="Highest high lookback window (default: 40)",
    )
    parser.add_argument(
        "--adx-period", type=int, default=14,
        help="ADX period (default: 14)",
    )
    parser.add_argument(
        "--sl-multiple", type=float, default=2.8,
        help="SL ATR multiple for Donchian lookback (default: 2.8)",
    )
    parser.add_argument(
        "--sl-period", type=int, default=10,
        help="SL ATR period for Donchian lookback (default: 10)",
    )
    parser.add_argument(
        "--tp-min-profit", dest="tp_min_profit_pct",
        type=float, default=0.2,
        help="TP minimum profit %% (default: 0.2)",
    )

    args = parser.parse_args()

    # ── Banner ──
    strat_name = {"sma": f"SMA{args.short_window} x SMA{args.long_window}",
                  "bb_adx": f"BB({args.bb_period},{args.bb_std}) + ADX({args.adx_period})",
                  "high_breakout": f"Highest({args.hh_window}) + ADX({args.adx_period})",
                  "basis_adx": f"Basis({args.bb_period}) + ADX({args.adx_period})"}[args.strategy]

    print(f"\n🚀 Backtester — {args.ticker}")
    print(f"   Strategy : {args.strategy} — {strat_name}")
    print(f"   Risk/trade : {args.risk_pct:.1f}% of equity")
    print(f"   Periode  : {args.start} — {args.end}")
    print(f"   Modal    : {get_currency(args.ticker, args.cash)} {args.cash:,.0f}")
    print()

    # 1. Download data
    df = fetch_data(args.ticker, args.start, args.end)

    # 2. Build strategy
    strategy_cls = build_strategy_class(args.strategy, args)

    # 3. Run backtest
    bt = Backtest(
        df,
        strategy_cls,
        cash=args.cash,
        commission=args.commission,
        finalize_trades=True,
    )
    stats = bt.run()
    print_stats(stats, initial_cash=args.cash, ticker=args.ticker)

    # 4. Optional optimisation
    if args.optimize:
        if args.strategy == "sma":
            print("  🔬 Optimasi parameter (short=5..25, long=20..55)...")
            result = bt.optimize(
                short_window=range(5, 30, 5),
                long_window=range(20, 60, 5),
                maximize="Sharpe Ratio",
                constraint=lambda p: p.short_window < p.long_window,
            )
            s = result["_strategy"]
            print(f"  ✅ Optimal  : SMA{s.short_window:.0f} x SMA{s.long_window:.0f}")
            print(f"     Sharpe  : {result['Sharpe Ratio']:.2f}  "
                  f"Return: {result['Return [%]']:.2f}%")
        elif args.strategy == "bb_adx":
            print("  🔬 Optimasi parameter BB+ADX (butuh waktu)...")
            result = bt.optimize(
                bb_period=[10, 20, 30],
                adx_period=[10, 14, 20],
                maximize="Sharpe Ratio",
            )
            s = result["_strategy"]
            print(f"  ✅ Optimal  : BB({s.bb_period:.0f},{s.bb_std:.1f}) + ADX({s.adx_period:.0f})")
            print(f"     Sharpe  : {result['Sharpe Ratio']:.2f}  "
                  f"Return: {result['Return [%]']:.2f}%")
        elif args.strategy == "high_breakout":
            print("  🔬 Optimasi parameter (HH=20..80, ADX=10..20)...")
            result = bt.optimize(
                hh_window=range(20, 85, 10),
                adx_period=[10, 14, 20],
                maximize="Sharpe Ratio",
            )
            s = result["_strategy"]
            print(f"  ✅ Optimal  : Highest({s.hh_window:.0f}) + ADX({s.adx_period:.0f})")
            print(f"     Sharpe  : {result['Sharpe Ratio']:.2f}  "
                  f"Return: {result['Return [%]']:.2f}%")
        print()

    # 5. Plot
    if args.save:
        out_dir = Path("reports")
        out_dir.mkdir(exist_ok=True)
        ticker_clean = args.ticker.replace(".", "_")
        plot_path = out_dir / f"{ticker_clean}_{args.strategy}.html"
        bt.plot(filename=str(plot_path), open_browser=False)
        print(f"  📈 Plot tersimpan: {plot_path}")
    else:
        print("  💡 Gunakan --save untuk simpan laporan HTML")

    print("\n✅ Selesai!\n")


if __name__ == "__main__":
    main()
