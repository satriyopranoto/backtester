# Backtester

Framework backtesting untuk strategi **Basis+ADX, Mean Reversion, Hybrid, Regime-Switching** pada saham US, IDX, forex, dan komoditas.

---

## 📦 Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

Dependencies: `backtesting`, `yfinance`, `numpy`, `pandas`, `matplotlib`

---

## 📥 Cara Tarik Data

Script **otomatis download data** dari Yahoo Finance saat dijalankan — tidak perlu download manual.

Contoh:
```python
# Di dalam script backtest
import yfinance as yf
df = yf.download('AAPL', period='10y', interval='1d', progress=False)
```

### Ticker Penting

| Pair | Ticker Yahoo |
|---|---|
| **Saham US** | `AAPL`, `MSFT`, `IBM`, `TSLA`, dll |
| **IDX (Indonesia)** | `BBCA.JK`, `BBRI.JK`, `BMRI.JK`, dll |
| **XAU/USD (Gold)** | `GC=F` |
| **GBP/USD** | `GBPUSD=X` |
| **EUR/USD** | `EURUSD=X` |
| **USD/CHF** | `USDCHF=X` |
| **DXY Index** | `DX-Y.NYB` |

### Ketersediaan Data per Timeframe

| Interval | Maks Data |
|---|---|
| **30m** | ~60 hari |
| **1h** | ~730 hari (2 tahun) |
| **1d** | Maksimal (1998+) |
| **1wk** | Maksimal |

---

## 🚀 Cara Menjalankan Backtest

### 1. Backtest Sederhana (Single Stock)

```bash
python run_ibm_basis_adx.py
```

Hasil output:
```
========================================
  IBM — Basis+ADX (1998-2026)
========================================
  Return   : +225.46%
  Sharpe   : 1.64
  Max DD   : -59.16%
  Trades   : 1968 | WR: 64.0%
```

### 2. Backtest CFD dengan Margin Call (XAU/USD)

```bash
python run_xau_cfd.py
```

Simulasi realistik dengan:
- Modal $1,000, Leverage 1:100
- Lot tetap (bisa diatur: 0.01, 0.02, 0.03, 0.05)
- **Margin Call / Stop-Out** di 30% Margin Level
- Margin otomatis dikembalikan saat posisi ditutup

### 3. Dynamic Portfolio (Multi Stock)

```bash
cd ../dynamicportfoliobacktester
python run_us_multi_tf_mod.py    # US 50 stocks
python run_id_intraday_1h.py     # IDX 116 stocks intraday
```

---

## 📊 Melihat Hasil

Hasil backtest langsung tercetak di terminal. Metrics yang ditampilkan:

| Metrik | Arti |
|---|---|
| **Return** | Total return dalam % |
| **CAGR** | Compound Annual Growth Rate |
| **Sharpe Ratio** | Return per unit volatilitas (>1 = baik) |
| **Max DD** | Maximum Drawdown (palung terbesar) |
| **Ulcer Index** | Root-mean-square drawdown (makin rendah makin baik) |
| **Martin Ratio** | CAGR / Ulcer Index (makin tinggi makin baik) |
| **Win Rate** | Persentase trade profit |
| **Profit Factor** | Total profit / total loss (>1 = profitable) |
| **MC** | Jumlah Margin Call (CFD only) |

### File Output

Beberapa script menghasilkan file CSV:
```
id_hybrid_equity.csv           → Equity curve IDX hybrid
screener_result.json           → Hasil screener (dashboard)
```

---

## 📋 Daftar Program Python

### Strategi Inti (folder `strategies/`)

| File | Strategi |
|---|---|
| `basis_adx_strategy.py` | **Basis+ADX** — trend following (ADX>20, +DI>-DI, Close>SMA20) |
| `mean_reversion_strategy.py` | **Mean Reversion** — RSI<30 + Close<LowerBB → Buy |
| `mean_rev_adx_strategy.py` | **Mean Rev ADX** — Mean Rev entry + ADX exit |
| `hybrid_basis_adx_strategy.py` | **Hybrid** — Basis+ADX entry + Mean Rev ADX exit |
| `regime_switching_strategy.py` | **Regime Switching** — trend_score menentukan strategi |
| `bb_adx_strategy.py` | **BB+ADX** — Bollinger Bands + ADX filter |
| `high_breakout_adx.py` | **High Breakout ADX** |
| `sma_crossover.py` | **SMA Crossover** |

### Single Stock Backtest

| File | Pair | Timeframe |
|---|---|---|
| `run_ibm_basis_adx.py` | IBM | 1d |
| `run_ibm_1h_fast.py` | IBM | **1h** (1998-2026, 50k bars) |
| `run_ibm_30m_fast.py` | IBM | **30m** (1998-2026, 92k bars) |
| `run_ibm_daily_fast.py` | IBM | **1d** |
| `run_ibm_1d_fixed_tp.py` | IBM | 1d fixed TP |
| `run_ibm_sma200_exit.py` | IBM | 1d SMA200 exit |
| `run_ibm_2h_daily_sl.py` | IBM | 2h |
| `resample_ibm.py` | IBM | Resample data |
| `run_tsla_basis_adx.py` | TSLA | 1d |
| `run_tsla_mean_rev_adx.py` | TSLA | 1d |

### Forex Backtest

| File | Pair | Timeframe |
|---|---|---|
| `run_gbpusd_30m.py` | GBP/USD | 30m |
| `run_gbpusd_multi_tf.py` | GBP/USD | 30m + Daily |
| `run_gbpusd_daily_multitf.py` | GBP/USD | Daily + Weekly |
| `run_gbpusd_mean_reversion.py` | GBP/USD | 1d Mean Rev |
| `run_gbpusd_mean_rev_adx.py` | GBP/USD | 1d Mean Rev ADX |
| `run_gbpusd_regime.py` | GBP/USD | 1d Regime |
| `run_eurusd_basis_adx.py` | EUR/USD | 1d |
| `run_eurusd_mean_reversion.py` | EUR/USD | 1d |
| `run_eurusd_mean_rev_adx.py` | EUR/USD | 1d |
| `run_eurusd_regime.py` | EUR/USD | 1d |
| `compare_eurusd.py` | EUR/USD | Perbandingan strategi |
| `compare_eurusd_30m_sl.py` | EUR/USD | 30m SL comparison |
| `compare_eurusd_all_tf.py` | EUR/USD | Multi timeframe |
| `run_usdchf_trend.py` | USD/CHF | 1h Basis+ADX |
| `run_usdchf_mr.py` | USD/CHF | 1h Mean Reversion |
| `run_usdchf_multitf.py` | USD/CHF | 1h + Daily |

### Komoditas

| File | Pair | Timeframe |
|---|---|---|
| `run_xau_cfd.py` | **XAU/USD (Gold)** | **1h CFD + MC** |
| `gold_all_tf.py` | XAU/USD | Multi TF |
| `compare_gc_f.py` | XAU/USD | Perbandingan |
| `run_gc_f_1h.py` | XAU/USD | 1h |
| `run_gc_f_valley.py` | XAU/USD | Valley detection |

### Analisis

| File | Fungsi |
|---|---|
| `calc_ibm_ulcer.py` | Hitung Ulcer Index & Martin Ratio |
| `check_ibm_price.py` | Cek harga historis IBM |
| `check_xau_data.py` | Cek ketersediaan data XAU/USD |
| `multi_tf_analysis.py` | Analisis multi-timeframe |
| `compare_all.py` | Bandingkan semua strategi |

### Lainnya

| File | Fungsi |
|---|---|
| `crypto_long_scanner.py` | Scanner crypto long |
| `check_btc_sl.py` | Cek SL Bitcoin |
| `scan_compatibility.py` | Scan kompatibilitas |
| `patch_adx_rising.py` | Patch ADX rising |

---

## 💡 Tips

1. **Single stock → intraday (1h/30m)** lebih untung karena banyak trade
2. **Portofolio → daily** lebih cocok (diversifikasi)
3. **FX ranging** → Mean Reversion > Basis+ADX
4. **Gold trending** → Basis+ADX Single TF optimal
5. **Multi-TF** berguna untuk intraday (filter daily), tidak perlu di daily
