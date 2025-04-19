import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

data = pd.read_csv('data/oil_combined_countries.csv', parse_dates=['gasDayStart'], index_col=['gasDayStart'])
# data['Date'] = data['Date'].ffill
data['net_flow'] = -1 * data['netWithdrawal']
data.sort_index(inplace=True)
print(data.sample(5))


data = data[data['name']=='Germany']
ts = data['net_flow']
result = seasonal_decompose(
    ts, 
    model='additive',
    filt=None,
    period=365,
    two_sided=True,
    extrapolate_trend=0
)

print(result)
result.plot()
plt.show()

trend = result.trend

trend_df = pd.DataFrame({
    'net_flow': ts,
    'trend': trend
})

trend_df.to_csv('data/germany_oil_trend.csv', index=False)
trend_df['brent_price_usd'] = data['brent_price_usd']

correlation = trend_df['trend'].corr(trend_df['brent_price_usd'])
print(f'Corr between germany gas flow trend and brent oil price: {correlation}')
