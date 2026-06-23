import json, urllib.request, urllib.parse

data = urllib.parse.urlencode({'ticker': 'BTC-USD'}).encode()
req = urllib.request.Request('http://localhost:5000/analyze', data=data)
resp = urllib.request.urlopen(req)
d = json.loads(resp.read())

price = d['last_price']
sl = d['last_sl']
rec = d['recommendation']
ta = d.get('trend_analysis', {})
c = ta.get('current', {})

print(f"BTC/USD saat ini:")
print(f"  Price: ${price:,.2f}")
print(f"  SL (Donchian): ${sl:,.2f}")
print(f"  Above SL? {'YES ✅' if price > sl else 'NO ❌'}")
print(f"  Jarak dari SL: {((price/sl) - 1) * 100:+.2f}%")
print(f"  Recommendation: {rec}")
print()
print(f"  SMA20: ${c.get('sma20',0):,.2f}")
print(f"  ADX: {c.get('adx')} | +DI: {c.get('pdi')} | -DI: {c.get('mdi')}")
print(f"  Above SMA20? {'YES' if c.get('above_sma20') else 'NO'}")
