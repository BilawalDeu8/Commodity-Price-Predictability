from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA models
for col in combined_data.columns:
    print(f"Fitting ARIMA for {col}:")
    arima_model = ARIMA(combined_data[col].dropna(), order=(1, 1, 1))  # Adjust p, d, q as needed
    arima_result = arima_model.fit()
    print(arima_result.summary())
    # Forecast
    forecast = arima_result.forecast(steps=10)  # Forecast 10 periods ahead
    print(f"Forecast for {col}:\n", forecast)
    
# Calculate RMSE and MSE for ARIMA
def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

def calculate_mse(actual, predicted):
    return mean_squared_error(actual, predicted)

# Example: ARIMA predictions
actual = combined_data['Coffee'].values[-10:]
predicted = forecast

print("RMSE for ARIMA:", calculate_rmse(actual, predicted))
print("MSE for ARIMA:", calculate_mse(actual, predicted))

# Plot the forecasted values for each commodity along with past month data
fig, ax = plt.subplots(3, 1, figsize=(10, 10))

for i, col in enumerate(combined_data.columns):
    ax[i].plot(combined_data.index[-30:], combined_data[col].values[-30:], label='Observed')
    ax[i].plot(pd.date_range(start=combined_data.index[-1], periods=10, freq='D'), forecast, label='Forecast')
    ax[i].set_title(f"{col} Forecast")
    ax[i].legend()
    
plt.tight_layout()
plt.show()

