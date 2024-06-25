# dataset https://finance.yahoo.com/quote/BTC-USD/history?p=BTC-USD
import os
from os import path
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import datetime
from datetime import datetime, date
from datetime import datetime, date, timedelta
import pandas_datareader as data
import pandas_datareader.data as web
import math
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Button
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplfinance as mpf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.layers import Reshape
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from pandas_datareader import data as pdr
import prediksi
from prediksi import prediksi_harga_bitcoin, display_prediction_results
from prediksi import calculate_moving_averages, calculate_vwap
import logging


def load_data(start_date, end_date):
    file_path = 'BTC-USD.csv'
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        df = yf.download('BTC-USD', start=start_date_str, end=end_date_str)
        df['Date'] = df.index
        df.reset_index(drop=True, inplace=True)
        df.to_csv(file_path, index=False)
    
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

def main():
    st.title('IMPLEMENTASI METODE LONG SHORT TERM MEMORY (LSTM) UNTUK MEMPREDIKSI HARGA BITCOIN')
    
    prediction_date = st.sidebar.date_input('Tanggal Prediksi', value=pd.to_datetime('2023-01-01'))
    prediction_date = pd.Timestamp(prediction_date)
    prediction_interval_days = st.sidebar.number_input('Interval Prediksi Hari', min_value=1, max_value=3000, value=30)
    history_length = 100
    
    start_date = pd.to_datetime('2014-09-17')
    end_date = pd.to_datetime('2023-12-31')
    data = load_data(start_date, end_date)
    data.set_index('Date', inplace=True)
    data = data[['Close', 'Volume']]
    
    data_start_date = data.index.min()
    data_end_date = data.index.max()
    
    if prediction_date < data_start_date or prediction_date > data_end_date:
        st.error(f'Tanggal yang ingin Anda prediksi ({prediction_date}) tidak ada pada dataset historis Bitcoin ({data_start_date} hingga {data_end_date}). Silakan unggah dataset baru dari Yahoo Finance.')
        uploaded_file = st.sidebar.file_uploader('Unggah dataset historis Bitcoin (CSV)', type=['csv'])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
            data = data[['Close', 'Volume']]
            data_start_date = data.index.min()
            data_end_date = data.index.max()
            
            data = calculate_moving_averages(data)
            data = calculate_vwap(data)
    
    if st.sidebar.button('Mulai Prediksi') and data_start_date <= prediction_date <= data_end_date:
        prediction_close, prediction_volume, scaled_data, scaler, y_close_test, y_volume_test, X_test = prediksi_harga_bitcoin(data, prediction_date, prediction_interval_days, history_length)
        display_prediction_results(data, prediction_close, prediction_volume, scaler, y_close_test, y_volume_test, X_test, prediction_date)

if __name__ == '__main__':
    main()

