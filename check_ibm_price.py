import yfinance as yf, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
df = yf.download('IBM', period='max', interval='1d', progress=False)
if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
c = df['Close'].values.astype(float)
df['year'] = df.index.year
yearly = df.groupby('year')['Close'].agg(['min','max'])
print('=== IBM Price by Year ===')
for yr in range(1998, 2006):
    if yr in yearly.index:
        lo = yearly.loc[yr,'min']
        hi = yearly.loc[yr,'max']
        chg = (hi-lo)/lo*100
        print(f"{yr}: Low=${lo:.2f} High=${hi:.2f} Range={chg:.1f}%")
min_idx = np.argmin(c)
print(f"\nLowest IBM: ${c[min_idx]:.2f} on {df.index[min_idx].date()}")
print(f"Highest IBM: ${c.max():.2f} on {df.index[c.argmax()].date()}")
