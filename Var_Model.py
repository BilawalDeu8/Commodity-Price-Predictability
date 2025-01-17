# Load the commodities data {Coffee, Crude Oil, Gold .csv}
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Load the data
coffee = pd.read_csv('/path/to/file')
crude_oil = pd.read_csv('/path/to/file')
gold = pd.read_csv('/path/to/file')

# Convert the first column to name to 'Price' for all datasets
for df in [coffee, crude_oil, gold]:
    df.columns = ['Date', 'Price']

# Convert Date columns to datetime and set as index
for df in [coffee, crude_oil, gold]:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

# Combine datasets on common dates
combined_data = pd.concat([coffee['Price'], crude_oil['Price'], gold['Price']], axis=1, join='inner')
combined_data.columns = ['Coffee', 'Crude_Oil', 'Gold']
combined_data.dropna(inplace=True)

# Apply differencing to make the data stationary
differenced_data = combined_data.diff().dropna()

# Check stationarity again after differencing

def check_stationarity(series):
    adf_result = adfuller(series)
    print(f"ADF Statistic: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")
    return adf_result[1] <= 0.05  # Stationary if p-value <= 0.05

for col in differenced_data.columns:
    print(f"Checking stationarity for {col} after differencing:")
    is_stationary = check_stationarity(differenced_data[col])
    print(f"{col} is {'stationary' if is_stationary else 'still non-stationary'}.")

# Update combined_data to the differenced version
combined_data = differenced_data
print(combined_data.head())

# This cell was written to address the datetime freq warning, but the predictions were worse off 
# Set frequency of the DateTimeIndex
combined_data.index = pd.to_datetime(combined_data.index)

# Assuming the data is daily, set the frequency to 'D'
combined_data = combined_data.asfreq('D')  # 'D' for daily frequency

# If your data is monthly, use:
# combined_data = combined_data.asfreq('M')  # 'M' for monthly frequency

# Backward fill missing values (use the next valid value for missing dates)
combined_data = combined_data.bfill()

print(combined_data.head())

from statsmodels.tsa.api import VAR

# Select optimal lag order
model = VAR(combined_data)
lag_selection = model.select_order(maxlags=15)
print("Lag order selection:\n", lag_selection.summary())

# Fit VAR model with optimal lag
optimal_lag = lag_selection.selected_orders['aic']  # Use AIC to choose lag
var_model = model.fit(optimal_lag)
print(var_model.summary())

# Impulse Response Function (IRF)
irf = var_model.irf(10)  # 10 periods ahead
irf.plot(orth=True)
plt.show()

# Granger causality tests
granger_test = var_model.test_causality('Coffee', 'Crude_Oil', kind='f')
print(granger_test.summary())

