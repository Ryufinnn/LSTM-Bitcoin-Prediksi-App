# dataset https://finance.yahoo.com/quote/BTC-USD/history?p=BTC-USD
import os
from os import path
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import datetime
from datetime import datetime, date, timedelta
import pandas_datareader as data
import pandas_datareader.data as web
import math
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.dates as mdates
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
import logging


def normalize_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def calculate_moving_averages(df):
    df['MA100_Close'] = df['Close'].rolling(window=100).mean()
    df['MA200_Close'] = df['Close'].rolling(window=200).mean()
    df['MA100_Volume'] = df['Volume'].rolling(window=100).mean()
    df['MA200_Volume'] = df['Volume'].rolling(window=200).mean()
    df = df.dropna()
    return df

def calculate_vwap(df):
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df

def calculate_rmse(true_values, predicted_values):
    return np.sqrt(mean_squared_error(true_values, predicted_values))

def calculate_mape(true_values, predicted_values):
    return mean_absolute_percentage_error(true_values, predicted_values)

def evaluate_prediction(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = calculate_rmse(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    mape = calculate_mape(true_values, predicted_values)
    return mse, rmse, mae, mape

def create_dataset(data, history_length):
    X, y_close, y_volume = [], [], []
    for i in range(history_length, len(data)):
        X.append(data[i-history_length:i])
        y_close.append(data[i, 0])
        y_volume.append(data[i, 1])
    return np.array(X), np.array(y_close), np.array(y_volume)

def prediksi_harga_bitcoin(data, prediction_date, prediction_interval_days, history_length):
    model_file_path = 'keras_model_500.h5'
    
    if os.path.exists(model_file_path):
        model = load_model(model_file_path)
        st.info('Menampilkan hasil prediksi.....')
    else:
        st.info('Model yang telah di latih tidak di temukan.')
        data = data[['Close', 'Volume']].values
        scaled_data, scaler = normalize_data(data)
        
        df = pd.DataFrame(data, columns=['Close', 'Volume'])
        df = calculate_moving_averages(df)
        df = calculate_vwap(df)
        
        X, y_close, y_volume = create_dataset(scaled_data, history_length)
        
        train_size = int(len(X) * 0.9)
        X_train, X_test = X[:train_size], X[train_size:]
        y_close_train, y_close_test = y_close[:train_size], y_close[train_size:]
        y_volume_train, y_volume_test = y_volume[:train_size], y_volume[train_size:]
        
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=60, activation='relu', return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(units=80, activation='relu', return_sequences=True))
        model.add(Dropout(0.6))
        model.add(LSTM(units=120, activation='relu'))
        model.add(Dropout(0.8))
        model.add(Dense(units=2))
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        model.fit(X_train, np.column_stack((y_close_train, y_volume_train)), epochs=500, batch_size=640)
        model.save(model_file_path)
        
    model = load_model(model_file_path)
    
    data = data[['Close', 'Volume']].values
    scaled_data, scaler = normalize_data(data)
    X, y_close, y_volume = create_dataset(scaled_data, history_length)
    
    predictions = model.predict(X[-prediction_interval_days:])
    prediction_close = predictions[:, 0]
    prediction_volume = predictions[:, 1]
    
    return prediction_close, prediction_volume, scaled_data, scaler, y_close[-prediction_interval_days:], y_volume[-prediction_interval_days:], X[-prediction_interval_days:]

def display_prediction_results(data, prediction_close, prediction_volume, scaler, y_close_test, y_volume_test, X_test, prediction_date):
    
    prediction_close = scaler.inverse_transform(np.concatenate((prediction_close.reshape(-1, 1), np.zeros((prediction_close.shape[0], 1))), axis=1))[:, 0]
    prediction_volume = scaler.inverse_transform(np.concatenate((np.zeros((prediction_volume.shape[0], 1)), prediction_volume.reshape(-1, 1)), axis=1))[:, 1]
    y_close_test = scaler.inverse_transform(np.concatenate((y_close_test.reshape(-1, 1), np.zeros((y_close_test.shape[0], 1))), axis=1))[:, 0]
    y_volume_test = scaler.inverse_transform(np.concatenate((np.zeros((y_volume_test.shape[0], 1)), y_volume_test.reshape(-1, 1)), axis=1))[:, 1]
    
    st.write("### Tabel Perbandingan Harga Asli dan Hasil Prediksi")
    results = pd.DataFrame({
        'Tanggal': pd.date_range(start=prediction_date, periods=len(prediction_close)),
        'Harga Asli (Close)': y_close_test,
        'Harga Prediksi (Close)': prediction_close,
        'Volume Asli': y_volume_test,
        'Volume Prediksi': prediction_volume
    })
    st.write(results)
    
    st.write("### Grafik Perbandingan Harga Asli (Close) dan Hasil Prediksi (Close)")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=results['Tanggal'], y=results['Harga Asli (Close)'], mode='lines', name='Harga Asli (Close)', line=dict(color='blue')), secondary_y=False)
    fig.add_trace(go.Scatter(x=results['Tanggal'], y=results['Harga Prediksi (Close)'], mode='lines', name='Harga Prediksi (Close)', line=dict(color='red')), secondary_y=False)
    fig.update_layout(title='Perbandingan Harga Asli dan Hasil Prediksi', xaxis_title='Tanggal', yaxis_title='Harga')
    st.plotly_chart(fig)
    
    st.write("### Grafik Hasil Prediksi Harga Asli (Close), Harga Prediksi (Close), Volume Asli dan Volume prediksi")
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    
    fig.add_trace(go.Scatter(x=results['Tanggal'], y=results['Harga Asli (Close)'], mode='lines', name='Harga Asli (Close)', line=dict(color='blue')), secondary_y=True)
    fig.add_trace(go.Scatter(x=results['Tanggal'], y=results['Harga Prediksi (Close)'], mode='lines', name='Harga Prediksi (Close)', line=dict(color='red')), secondary_y=True)

    
    fig.add_trace(go.Scatter(x=results['Tanggal'], y=results['Volume Asli'], mode='lines', name='Volume Asli', line=dict(color='green')), secondary_y=False)
    fig.add_trace(go.Scatter(x=results['Tanggal'], y=results['Volume Prediksi'], mode='lines', name='Volume Prediksi', line=dict(color='red')), secondary_y=False)

    
    fig.update_layout(
        title='Perbandingan Harga dan Volume Asli dengan Hasil Prediksi',
        xaxis_title='Tanggal',
        yaxis=dict(title='Volume', side='left'),
        yaxis2=dict(title='Harga', overlaying='y', side='right')
    )

    st.plotly_chart(fig)
    
    
    latest_date = results['Tanggal'].iloc[-1]
    interval_prediksi_hari = len(results)

    st.write(f"""
    ### Keterangan:
    Dari grafik Chart di atas hasil prediksi harga bitcoin pada tanggal {prediction_date.strftime('%Y-%m-%d')} dengan interval prediksi {interval_prediksi_hari} hari menunjukkan hasil:
    """)

    
    keterangan = pd.DataFrame({
        'Keterangan': ['Harga Asli (Close)', 'Harga Prediksi (Close)', 'Volume Asli', 'Volume Prediksi'],
        'Nilai': [
            f"{results['Harga Asli (Close)'].iloc[-1]:.2f}",
            f"{results['Harga Prediksi (Close)'].iloc[-1]:.2f}",
            f"{results['Volume Asli'].iloc[-1]:,.0f}",
            f"{results['Volume Prediksi'].iloc[-1]:,.0f}"
        ]
    })
    
    st.table(keterangan)
    
    mse_close, rmse_close, mae_close, mape_close = evaluate_prediction(y_close_test, prediction_close)
    
    mse_volume, rmse_volume, mae_volume, mape_volume = evaluate_prediction(y_volume_test, prediction_volume)
    
    st.write("## Evaluasi Model Prediksi")
    
    eval_close_df = pd.DataFrame({
        'Metrik': ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)', 'Mean Absolute Percentage Error (MAPE)'],
        'Nilai': [f"{mse_close:.4f}", f"{rmse_close:.4f}", f"{mae_close:.4f}", f"{mape_close:.2%}"]
    })
    
    st.write("### Harga Penutupan (Close):")
    st.table(eval_close_df)
    
    eval_volume_df = pd.DataFrame({
        'Metrik': ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)', 'Mean Absolute Percentage Error (MAPE)'],
        'Nilai': [f"{mse_volume:.4f}", f"{rmse_volume:.4f}", f"{mae_volume:.4f}", f"{mape_volume:.2%}"]
    })
    
    st.write("### Volume:")
    st.table(eval_volume_df)
    
    st.write("""
    ### Keterangan:
    Presentase hasil prediksi yang di tampilkan menggunakan MAPE dengan presentase terkecil untuk variabel (Close) dan Variabel (Volume).
    """)
    
    st.write("## Penjelasan Hasil Prediksi")
    st.write("""
    Model LSTM dilatih dengan data historis harga dan volume Bitcoin. Data ini diolah melalui beberapa tahapan:
    1. Normalisasi: Data dinormalisasi menggunakan `MinMaxScaler` untuk mempercepat dan meningkatkan akurasi pelatihan model.
    2. Moving Averages: Rata-rata bergerak (100 dan 200 hari) dihitung untuk harga penutupan dan volume.
    3. VWAP: Volume Weighted Average Price dihitung.
    4. Pembuatan Dataset: Dataset dibuat dengan panjang data historis (history_length) yang telah ditentukan.
    """)
    
    st.write("""
    ### Penjelasan Rumus Evaluasi Model
    - **Mean Squared Error (MSE)**: MSE mengukur rata-rata kuadrat selisih antara nilai yang diprediksi dan nilai asli. MSE dihitung dengan rumus:
    """)
    st.latex(r''' \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 ''')
    st.write("""
    dimana \( y_i \) adalah nilai asli, \( \hat{y}_i \) adalah nilai prediksi, dan \( n \) adalah jumlah data.
    
    - **Root Mean Squared Error (RMSE)**: RMSE adalah akar kuadrat dari MSE. RMSE memberikan ukuran error yang lebih dapat dimengerti karena memiliki satuan yang sama dengan data asli. RMSE dihitung dengan rumus:
    """)
    st.latex(r''' \text{RMSE} = \sqrt{\text{MSE}} ''')
    st.write("""
    - **Mean Absolute Error (MAE)**: MAE mengukur rata-rata selisih absolut antara nilai yang diprediksi dan nilai asli. MAE dihitung dengan rumus:
    """)
    st.latex(r''' \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| ''')
    st.write("""
    - **Mean Absolute Percentage Error (MAPE)**: MAPE mengukur rata-rata persentase kesalahan absolut antara nilai yang diprediksi dan nilai asli. MAPE dihitung dengan rumus:
    """)
    st.latex(r''' \text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| ''')
    
    save_prediction_results(data, prediction_close, prediction_volume, y_close_test, y_volume_test, prediction_date)
    
def save_prediction_results(data, prediction_close, prediction_volume, y_close_test, y_volume_test, prediction_date):
    results_folder = 'hasil_prediksi'
    os.makedirs(results_folder, exist_ok=True)
    
    prediction_dates = pd.date_range(start=prediction_date, periods=len(prediction_close))
    prediction_results = pd.DataFrame({
        'Date': prediction_dates,
        'Close_Actual': y_close_test,
        'Close_Predicted': prediction_close,
        'Volume_Actual': y_volume_test,
        'Volume_Predicted': prediction_volume
    })
    
    prediction_date_str = prediction_date.strftime('%Y-%m-%d')
    filename = f'Hasil_prediksi_{prediction_date_str}.csv'
    prediction_results.to_csv(os.path.join(results_folder, filename), index=False)
    st.success(f'Hasil prediksi telah disimpan di folder hasil_prediksi dengan nama file {filename}.')

