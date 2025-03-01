import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM

st.title("📈 Stock Market Prediction with LSTM")
st.sidebar.header("Select a Stock")

companies = {
    "Apple (AAPL)": "AAPL",
    "Google (GOOG)": "GOOG",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN"
}
selected_company = st.sidebar.selectbox("Choose a company:", list(companies.keys()))
symbol = companies[selected_company]

current_year = datetime.now().year
years = list(range(2000, current_year + 1)) 
start_year = st.sidebar.selectbox("Select the starting year:", years, index=years.index(2012))  

start_date = f"{start_year}-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")
df = yf.download(symbol, start=start_date, end=end_date)

if 'Close' not in df.columns:
    st.error("'Close' column not found in the data")
    st.stop()

data = df[['Close']].copy()
dataset = data.values
training_data_len = int(np.ceil(len(dataset) * 0.95))

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[:training_data_len]
x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model_filename = f"{symbol}_model.h5"

if os.path.exists(model_filename):
    st.sidebar.success(f"Loading pre-trained model for {symbol}...")
    model = load_model(model_filename)
else:
   
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        LSTM(64, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    st.sidebar.write("Training Model... (This may take a few minutes)")
    model.fit(x_train, y_train, batch_size=1, epochs=5, verbose=0)
    model.save(model_filename)  
    st.sidebar.success("Model Trained & Saved!")

test_data = scaled_data[training_data_len - 60:]
x_test, y_test = [], dataset[training_data_len:]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
st.sidebar.write(f"RMSE: {rmse:.2f}")

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

st.subheader("📊 Stock Price Prediction")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(train['Close'], label='Train Data')
ax.plot(valid[['Close', 'Predictions']], label=['Actual', 'Predicted'])
ax.legend()
st.pyplot(fig)

st.write("### 📌 Stock Data")
st.dataframe(df.tail())

