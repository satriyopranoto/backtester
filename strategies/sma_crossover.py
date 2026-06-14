"""SMA Crossover Strategy — strategi klasik untuk contoh backtest."""

from backtesting import Backtest, Strategy
from backtesting.lib import crossover

import pandas as pd


class SmaCrossover(Strategy):
    """Simple Moving Average crossover strategy.

    Beli ketika SMA cepat (short) memotong SMA lambat (long) dari bawah ke atas.
    Jual ketika sebaliknya.

    Parameters
    ----------
    short_window : int
        Periode SMA cepat (default: 10).
    long_window : int
        Periode SMA lambat (default: 30).
    """

    short_window = 10
    long_window = 30

    def init(self):
        price = self.data.Close
        self.sma_short = self.I(
            lambda x: pd.Series(x).rolling(self.short_window).mean(),
            price,
            name=f"SMA{self.short_window}",
            overlay=True,
        )
        self.sma_long = self.I(
            lambda x: pd.Series(x).rolling(self.long_window).mean(),
            price,
            name=f"SMA{self.long_window}",
            overlay=True,
        )

    def next(self):
        """Dipanggil setiap baris data."""
        if crossover(self.sma_short, self.sma_long):
            self.buy()
        elif crossover(self.sma_long, self.sma_short):
            self.sell()
