from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(combined_data)

# Prepare data for LSTM
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 10
X, y = create_sequences(scaled_data, sequence_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model_lstm = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(sequence_length, X.shape[2])),
    tf.keras.layers.Dense(3)  # 3 commodities
])
model_lstm.compile(optimizer='adam', loss='mse')

# Train LSTM model
history = model_lstm.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Evaluate LSTM model
lstm_predictions = model_lstm.predict(X_test)
predicted_prices = scaler.inverse_transform(lstm_predictions)

# Display predictions for the first few samples
print("LSTM Predictions:\n", predicted_prices[:5])

# Plot training history
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('LSTM Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# LSTM predictions for the next month (10 periodss)
lstm_forecast = model_lstm.predict(np.expand_dims(scaled_data[-sequence_length:], axis=0))
lstm_forecast = scaler.inverse_transform(lstm_forecast)
print("LSTM Forecast:\n", lstm_forecast)

