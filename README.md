# Backtester

Simple backtesting tool using [Backtesting.py](https://kernc.github.io/backtesting.py/).

## Setup

```bash
python -m venv venv
source venv/Scripts/activate   # Windows (git-bash)
# atau: venv\Scripts\activate   # Windows cmd

pip install -r requirements.txt
```

## Usage

```bash
python main.py --ticker BBCA.JK --start 2022-01-01 --end 2024-12-31
```

## Structure

```
backtester/
├── strategies/     # Trading strategies
│   └── sma_crossover.py
├── data/           # CSV data cache
├── reports/        # Output HTML reports
├── main.py         # Entry point
└── requirements.txt
```
