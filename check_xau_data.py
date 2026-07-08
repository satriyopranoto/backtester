import yfinance as yf, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

# Data availability
print("=== XAU/USD 1h Data Availability ===")
for p in ['1mo','3mo','6mo','1y','2y']:
    df = yf.download('GC=F', period=p, interval='1h', progress=False)
    if not df.empty and len(df) > 10:
        print(f"  {p}: {len(df)} bars | {df.index[0]} → {df.index[-1]}")
    else:
        print(f"  {p}: No data")

# Current price
df = yf.download('GC=F', period='5d', interval='1h', progress=False)
if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
price = df['Close'].iloc[-1]
print(f"\nGold price: ${price:.2f}")

# CFD calc
leverage = 100
lot_size_gold = 100  # standard lot = 100 oz
contract_size = lot_size_gold * 0.01  # 0.01 lot = 1 oz
margin_per_lot = (price * contract_size) / leverage

print(f"\n=== CFD Simulasi ===")
print(f"Modal: $1,000")
print(f"Leverage: 1:{leverage}")
print(f"Lot: 0.01 (1 oz)")
print(f"Margin per posisi: ${margin_per_lot:.2f}")
print(f"Maks posisi: {int(1000/margin_per_lot)}")
print(f"1 pip gold = $10 per lot standard, $0.1 per 0.01 lot")
print(f"SL 200 pips = $20 loss per 0.01 lot")
