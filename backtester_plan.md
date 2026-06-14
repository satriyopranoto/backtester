# Plan: BB Breakout + ADX Strategy for Backtester

## Source Strategy (dari stocktrade `app.py`)
- **SL**: Donchian Channel SL (`calculate_sl`)
- **BB**: Bollinger Bands (20,2) — `calculate_bollinger_bands`
- **ADX**: Wilder's smoothed ADX, +DI (PDI), -DI (MDI) — `calculate_adx`
- **BB Screener buy logic**: close > upper_bb AND price > sl AND adx > 25 AND adx_rising AND pdi > mdi

## Strategy Rules

### Entry (BUY) — semua harus TRUE:
1. `Close > Upper Bollinger Band` — breakout
2. `Low > SL` — harga tidak di bawah stop loss
3. `ADX > 25` — tren kuat
4. `ADX > ADX[5]` (ADX rising, 5 bars ago) — tren menguat
5. `PDI > MDI` — arah bullish

### SL (Stop Loss) — Donchian Channel:
- `ero = int(2.8 * 10)` → 28 bar lookback
- `r = highest(high, ero)`, `s = lowest(low, ero)` 
- `ab = high > r[1] ? 1 : (low < s[1] ? -1 : 0)`
- `ac = valuewhen(ab != 0, ab, 0)` — persistent direction via ffill
- `sl = ac == 1 ? s : r` — bullish use low channel, bearish use high channel

### Cut Loss:
- `cut_loss_point = valuewhen(buy, sl)` — SL value *pada saat entry* (fixed)
- Exit when: `Close < cut_loss_point`

### Take Profit:
- `Floating Profit > 0.2%` AND `Close < current SL`
- Artinya: kita sudah profit minimal 0.2%, lalu harga pullback ke bawah trailing SL

### Exit Rules:
- **Hanya** exit via Cut Loss atau Take Profit — tidak ada exit lain
- Tidak ada time-based exit, tidak ada reverse signal exit

## Files to Create/Modify

### 1. `strategies/bb_adx_strategy.py` (NEW)
- Class `BbAdxStrategy` extends `Strategy`
- `init()`: calculate BB, ADX, PDI, MDI, SL sebagai indicator series
- `next()`: entry + exit logic

### 2. `main.py` (MODIFY)
- Add `--strategy` argument to choose between `sma` and `bb_adx`
- Add strategy-specific params: `--bb-period`, `--bb-std`, `--adx-period`, `--sl-multiple`, `--sl-period`

### 3. Untuk ADX, PDI, MDI di `init()`:
- ADX indicator perlu di-calculate manual pake numpy/pandas di `init()` via `self.I()`
- Problem: `self.I()` di Backtesting.py hanya terima fungsi yang return array 1D
- Solusi: bungkus setiap series (ADX, PDI, MDI) dalam `self.I()` terpisah

### 4. Untuk SL di `init()`:
- Fungsi `calculate_sl()` juga perlu dijadikan indicator via `self.I()`

### 5. Entry/Exit State:
- Simpan `self.entry_sl = None` untuk cut loss point
- `self.in_position = False` tracker
- Di `next()`: cek entry conditions → buy, atau cek exit conditions → sell

### 6. Take Profit Check:
- `self.equity` / `self.data.Close` based floating profit calculation
- Atau pake `self.position.pl_pct` (Backtesting.py built-in)

## Considerations / Pitfalls
- **Backtesting.py short selling**: `self.sell()` opens a short. Kita exit long dengan `self.position.close()`. 
- **Margin warnings**: BBCA harga tinggi, perlu `self.buy(size=...)` yang wajar
- **ADX rising check**: `adx > adx[5]` — pake .shift(5) atau simpan array-nya
- **Low > SL**: pastikan SL dihitung dari bar sebelumnya (tidak look-ahead bias)
- **Close < SL**: ganti ke `self.data.Close[-1] < self.sl[-1]` untuk current bar
- **Floating profit**: `self.position.pl_pct` untuk unrealized P&L %

## Test Plan
1. Test dengan BBCA.JK (2024 only — fast)
2. Cek entry/exit timing
3. Bandingkan hasil dengan BB Screener output
4. Optimasi parameter ADX period, SL lookback

## Next Steps (after plan approved)
1. Create `strategies/bb_adx_strategy.py` 
2. Update `main.py` with strategy selector
3. Test run
4. Debug & fix
