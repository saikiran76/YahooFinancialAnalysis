import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas_datareader as web
import datetime
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st

start = datetime.datetime(2020, 1, 1)
end = datetime.datetime(2023, 1, 1)

st.title('Finance Stock Trend Prediction')


user_input = st.text_input('Enter Stock Ticker', 'AAPL')

try:
    df = yf.download(user_input, start=start, end=end)
except ValueError:
    st.write("Error: No data available for the entered stock symbol.")
    exit()
# df = yf.download(user_input, start=start, end=end)

st.subheader('Data from 2020 to 2023')
st.write(df.describe())

df.index = pd.Series(df.index).dt.strftime('%Y-%m-%d')

fig = px.line(df, x=df.index, y='Close')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Close Price')
st.plotly_chart(fig)

st.subheader('Closing Price vs Time with Moving Average of 100')
ma100 = df['Close'].rolling(window=100).mean()
fig = px.line(df, x=df.index, y='Close', title='Closing Price vs Time with Moving Average of 100')
fig.add_scatter(x=df.index, y=ma100, name='Moving Average')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Close Price')
st.plotly_chart(fig)

st.subheader('Closing Price vs Time with Moving Average of 200')
ma200 = df['Close'].rolling(window=200).mean()
fig = px.line(df, x=df.index, y='Close', title='Closing Price vs Time with Moving Average of 200')
fig.add_scatter(x=df.index, y=ma200, name='Moving Average')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Close Price')
st.plotly_chart(fig)

# Combined plot with 2 moving averages
fig = px.line(df, x=df.index, y='Close', title='Closing Price vs Time with Moving Average of 100 and 200')
fig.add_scatter(x=df.index, y=ma100, name='Moving Average 100')
fig.add_scatter(x=df.index, y=ma200, name='Moving Average 200')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Close Price')
st.plotly_chart(fig)

st.subheader('Closing Price vs Time with Exponential Moving Average of 100')
ema100 = df['Close'].ewm(span=100, adjust=False).mean()
fig = px.line(df, x=df.index, y='Close', title='Closing Price vs Time with Exponential Moving Average of 100')
fig.add_scatter(x=df.index, y=ema100, name='Exponential Moving Average')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Close Price')
st.plotly_chart(fig)







# Data Preparation
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

# Scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_arr = scaler.fit_transform(data_training)

# Model Building
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(data_training_arr, order=(1,1,1), seasonal_order=(1,1,1,12))
model_fit = model.fit()

# Data Preparation for Prediction
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

# Making Predictions
x_test = input_data[:-1]
y_test = input_data[-1]

y_predicted = model_fit.predict(start=100, end=len(input_data)-1)

# Rescaling Predictions
# ===================

y_predicted = scaler.inverse_transform(y_predicted.reshape(-1,1))[:,0]
y_test = scaler.inverse_transform(y_test.reshape(-1,1))[:,0]

# Plotting Predictions
# ===================

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(plt)