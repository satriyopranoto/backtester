"""
Backtest Basis+ADX Population Strategy — EUR/USD Daily, last 2 years.

Master's strategy:
  Entry (population filters):
    - In last 100 bars, ADX > 25 for >= 35 bars
    - In last 100 bars, Close > SMA20 for >= 35 bars
    - Current bar: Low > Donchian SL
    - Current bar: Close > Basis (SMA20)
    - Current bar: ADX > 25
    - Current bar: +DI > -DI
    - Current bar: ADX rising (ADX > ADX[5])
    - Current bar: +DI rising (PDI > PDI[5])

  Exit:
    - Cut Loss: Close < entry_sl
    - Take Profit: floating > 0.4 x stop_dist% AND Close < current SL
"""
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from backtesting import Backtest, Strategy
from strategies.bb_adx_strategy import donchian_sl, calc_adx


class BasisAdxPopulationStrategy(Strategy):
    """Basis + ADX with population-based filter (Master's strategy)."""

    bb_period = 20
    adx_period = 14
    sl_multiple = 2.8
    sl_period = 10
    risk_pct = 1.0

    # Population parameters (Master's specs)
    population_bars = 100      # lookback window
    adx_threshold = 25         # ADX > 25
    min_adx_pct = 35           # at least 35% of bars
    min_sma_pct = 35           # at least 35% of bars

    _entry_sl = None
    _entry_price = None
    _tp_threshold_pct = None

    def init(self):
        def _sma(arr, period):
            return pd.Series(arr).rolling(period).mean().values

        # Basis = SMA20
        self.basis = self.I(
            _sma, self.data.Close, self.bb_period,
            name=f"SMA({self.bb_period})", overlay=True,
        )

        # Donchian SL
        self.sl = self.I(
            donchian_sl, self.data.High, self.data.Low,
            self.sl_multiple, self.sl_period,
            name="SL (Donchian)", overlay=True,
        )

        # ADX / PDI / MDI
        self.adx_arr, self.pdi_arr, self.mdi_arr = calc_adx(
            np.asarray(self.data.High),
            np.asarray(self.data.Low),
            np.asarray(self.data.Close),
            self.adx_period,
        )
        self.I(
            lambda: pd.DataFrame(
                {"ADX": self.adx_arr, "+DI": self.pdi_arr, "-DI": self.mdi_arr}
            ),
            name="ADX/DI", overlay=False,
        )

        # Pre-compute SMA20 for population check
        self.sma20 = _sma(np.asarray(self.data.Close), self.bb_period)

    def next(self):
        idx = len(self.data) - 1

        close = float(self.data.Close[-1])
        low = float(self.data.Low[-1])
        high = float(self.data.High[-1])
        basis = float(self.basis[-1])
        sl = float(self.sl[-1])

        adx = float(self.adx_arr[idx])
        pdi = float(self.pdi_arr[idx])
        mdi = float(self.mdi_arr[idx])
        adx_5ago = float(self.adx_arr[idx - 5]) if idx >= 5 else 0.0
        pdi_5ago = float(self.pdi_arr[idx - 5]) if idx >= 5 else 0.0
        is_nan = np.isnan(adx) or np.isnan(pdi) or np.isnan(mdi)

        # ── POPULATION FILTER (over last N bars) ────────────
        pop_ok = False
        if idx >= self.population_bars:
            start = idx - self.population_bars + 1
            adx_slice = self.adx_arr[start:idx + 1]
            sma_slice = self.sma20[start:idx + 1]
            close_slice = np.asarray(self.data.Close)[start:idx + 1]

            adx_gt_25 = np.sum(adx_slice > self.adx_threshold)
            close_gt_sma = np.sum(close_slice > sma_slice)

            adx_pct = (adx_gt_25 / self.population_bars) * 100
            sma_pct = (close_gt_sma / self.population_bars) * 100

            pop_ok = (adx_pct >= self.min_adx_pct and
                      sma_pct >= self.min_sma_pct)

        # ─────────────────────────────────────────────────────
        #  1. EXIT
        # ─────────────────────────────────────────────────────
        if self._entry_sl is not None and self._entry_price is not None:
            # Cut Loss
            if close < self._entry_sl:
                self.position.close()
                self._entry_sl = None
                self._entry_price = None
                self._tp_threshold_pct = None
                return

            # Take Profit
            tp_threshold = self._tp_threshold_pct if self._tp_threshold_pct is not None else 0.2
            if self._entry_price > 0 and tp_threshold is not None:
                floating_pct = ((close - self._entry_price) / self._entry_price) * 100.0
                if floating_pct > tp_threshold and close < sl:
                    self.position.close()
                    self._entry_sl = None
                    self._entry_price = None
                    self._tp_threshold_pct = None
                    return
            return

        # ─────────────────────────────────────────────────────
        #  2. ENTRY
        # ─────────────────────────────────────────────────────
        if (pop_ok
            and low > sl
            and close > basis
            and not is_nan
            and adx > self.adx_threshold
            and adx > adx_5ago
            and pdi > mdi
            and pdi > pdi_5ago
        ):
            stop_dist = abs(close - sl)
            if stop_dist > 0:
                risk_amount = self.equity * (self.risk_pct / 100.0)
                size = int(risk_amount / stop_dist)
                max_by_cash = int((self.equity * 0.95) / close)
                if max_by_cash >= 1:
                    size = max(1, size)
                size = min(size, max_by_cash)
            else:
                size = 0

            if size > 0:
                self.buy(size=size)
                self._entry_sl = sl
                self._entry_price = close
                self._tp_threshold_pct = (stop_dist / close) * 100.0 * 0.4


# ── Data ──
print("=" * 60)
print("  EUR/USD Daily — Population Basis+ADX Backtest")
print("  (ADX>25 >=35% & Close>SMA20 >=35% of last 100 bars)")
print("=" * 60)

ticker = "EURUSD=X"
print(f"\nDownloading {ticker} daily...")
data = yf.download(ticker, period="2y", interval="1d", auto_adjust=True)

# Flatten multi-level columns if needed
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = data.dropna(subset=["Open", "High", "Low", "Close"])
print(f"  Bars: {len(data)} ({data.index[0].date()} — {data.index[-1].date()})")
print(f"  Range: {data['Low'].min():.4f} — {data['High'].max():.4f}")

# ── Backtest ──
bt = Backtest(data, BasisAdxPopulationStrategy, cash=100_000, commission=0.0001, finalize_trades=True)
stats = bt.run()

print("\n" + "─" * 60)
print("  BACKTEST RESULTS")
print("─" * 60)
print(f"  Return        : {stats['Return [%]']:.2f}%")
print(f"  Equity Final  : ${stats['Equity Final [$]']:,.2f}")
print(f"  Sharpe Ratio  : {stats['Sharpe Ratio']:.2f}")
print(f"  Max DD        : {stats['Max. Drawdown [%]']:.2f}%")
print(f"  # Trades      : {stats['# Trades']}")
print(f"  Win Rate      : {stats.get('Win Rate [%]', 0):.1f}%")
print(f"  Best Trade    : {stats.get('Best Trade [%]', 0):.2f}%")
print(f"  Worst Trade   : {stats.get('Worst Trade [%]', 0):.2f}%")
print(f"  Avg Trade     : {stats.get('Avg Trade [%]', 0):.2f}%")

# Show individual trades
print("\n" + "─" * 60)
print("  TRADE LIST")
print("─" * 60)
trades = stats['_trades']
if trades is not None and len(trades) > 0:
    for i, trade in trades.iterrows():
        print(f"  {trade['EntryTime'].date() if hasattr(trade['EntryTime'], 'date') else trade['EntryTime']} "
              f"→ {trade['ExitTime'].date() if hasattr(trade['ExitTime'], 'date') else trade['ExitTime']} "
              f"| PnL: {trade['PnL']:+.2f} | Return: {trade.get('ReturnPct', 0)*100:.2f}%")
else:
    print("  No trades")

print("\n" + "=" * 60)
print("  ✅ Done")
print("=" * 60)

# Save report
report_dir = Path("/c/Users/Acer/code/backtester/reports")
report_dir.mkdir(exist_ok=True)
bt.plot(filename=str(report_dir / "EURUSD_basis_adx_population_daily.html"), open_browser=False)
print(f"\n  Report saved: reports/EURUSD_basis_adx_population_daily.html")
