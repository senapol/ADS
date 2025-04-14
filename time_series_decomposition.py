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
