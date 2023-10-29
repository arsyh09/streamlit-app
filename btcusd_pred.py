import streamlit as st
from binance.client import Client
import pandas as pd
import pandas_ta as ta
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime
import plotly.express as px
import os

# Mendapatkan nilai dari variabel lingkungan
api_key = os.environ.get('MY_API_KEY')
api_secret = os.environ.get('MY_API_SECRET')


# Fungsi untuk mengambil data dari Binance API sesuai timeframe
def get_binance_data(api_key, api_secret, selected_interval):
    if selected_interval == '1 menit':
        interval = Client.KLINE_INTERVAL_1MINUTE
    elif selected_interval == '5 menit':
        interval = Client.KLINE_INTERVAL_5MINUTE
    elif selected_interval == '15 menit':
        interval = Client.KLINE_INTERVAL_15MINUTE
    else:
        raise ValueError('Timeframe tidak valid.')
    client = Client(api_key, api_secret)
    symbol = 'BTCUSDT'
    klines = client.get_historical_klines(symbol, interval)

    # Konversi ke DataFrame
    latest_data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 
                                                'volume', 'close_time', 'quote_asset_volume', 
                                                'number_of_trades', 'taker_buy_base_asset_volume', 
                                                'taker_buy_quote_asset_volume', 'ignore'])

    # Convert timestamp to datetime format
    latest_data['timestamp'] = pd.to_datetime(latest_data['timestamp'], unit='ms')

    # Set timestamp as index
    latest_data.set_index('timestamp', inplace=True)

    # Hapus kolom yang tidak diperlukan
    latest_data.drop(['volume', 'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                        'ignore'], axis=1, inplace=True)

    # Konversi tipe data kolom ke float
    latest_data = latest_data.astype(float)

    # Hitung SMA
    latest_data['sma5'] = ta.sma(latest_data['close'], length=5)

    # Hitung EMA
    latest_data['ema5'] = ta.ema(latest_data['close'], length=5)

    # Hitung WMA
    latest_data['wma5'] = ta.wma(latest_data['close'], length=5)

    return latest_data


# Fungsi untuk memprediksi harga dengan model ELM
def predict_price(data, selected_interval):
    # Muat model sesuai dengan timeframe yang dipilih
    if selected_interval == '1 menit':
        with open('ELM1m_pkl', 'rb') as model_file:
            elm_model = pickle.load(model_file)
    elif selected_interval == '5 menit':
        with open('ELM5m_pkl', 'rb') as model_file:
            elm_model = pickle.load(model_file)
    elif selected_interval == '15 menit':
        with open('ELM15m_pkl', 'rb') as model_file:
            elm_model = pickle.load(model_file)
    else:
        raise ValueError('Timeframe tidak valid.')
    
    # data terbaru di normalisasi
    scaler = MinMaxScaler()
    data_load = data[-100:-1]
    #data_load = data[-100:]
    data_scaled = scaler.fit_transform(data_load)

    # Hitung output dari model ELM
    y_pred = elm_model.predict(data_scaled)
    
    # Denormalisasi output
    y_pred_denormalized = ((y_pred * (data_load['close'].max() - data_load['close'].min())) + data_load['close'].min()).round(2)

    # Ambil prediksi harga Bitcoin terbaru
    return y_pred_denormalized[-1][0]

# Fungsi yang akan dijalankan dengan interval waktu
def scheduler(selected_interval):
    date_pred = pd.DataFrame(columns=["Datetime", "Prediksi"])
    aktual_selisih = pd.DataFrame(columns=["Aktual", "Selisih Harga"])
    header = st.empty()
    placeholder = st.empty()

    while True:
        # mendapatkan waktu saat ini
        now = datetime.now()

        if selected_interval == '1 menit' and now.second == 10:
            current_time = now.strftime("%Y-%m-%d %H:%M")
            # Memanggil get_binance_data dengan interval yang sesuai
            data = get_binance_data(api_key, api_secret, selected_interval)
            # prediksi harga dari data pada fungsi get_data
            prediction = predict_price(data, selected_interval)

            date_pred.loc[len(date_pred)] = {"Datetime": current_time, "Prediksi": prediction}

            with header.container():
                kpi1, kpi2 = st.columns(2)
                kpi1.metric(label="Datetime", value=date_pred.iloc[-1]["Datetime"])
                kpi2.metric(label="Prediksi (USD)", value=date_pred.iloc[-1]["Prediksi"])

            time.sleep(47)

            data_aktual = get_binance_data(api_key, api_secret, selected_interval)
            harga_aktual = data_aktual['close'].iloc[-1]

            selisih_harga = prediction - harga_aktual

            # Menambahkan data baru ke dataframe
            aktual_selisih.loc[len(aktual_selisih)] = {"Aktual": harga_aktual, "Selisih Harga": selisih_harga}

            # menggabungkan data
            result = pd.concat([date_pred, aktual_selisih], axis=1)

            with placeholder.container():
                # menampilkan dataframe
                st.dataframe(data=result.iloc[-5:], use_container_width=True, hide_index=True)

                # menampilkan chart
                fig = px.line(result.iloc[-20:], x="Datetime", y=["Prediksi", "Aktual"], title="Chart Harga")
                fig.update_yaxes(title_text='Harga (USD)')
                st.write(fig)

        elif selected_interval == '5 menit' and now.second == 10 and now.minute % 5 == 0:
            current_time = now.strftime("%Y-%m-%d %H:%M")
            # Memanggil get_binance_data dengan interval yang sesuai
            data = get_binance_data(api_key, api_secret, selected_interval)
            # prediksi harga dari data pada fungsi get_data
            prediction = predict_price(data, selected_interval)

            # menambahkan data baru
            date_pred.loc[len(date_pred)] = {"Datetime": current_time, "Prediksi": prediction}

            with header.container():
                kpi1, kpi2 = st.columns(2)
                kpi1.metric(label="Datetime", value=date_pred.iloc[-1]["Datetime"])
                kpi2.metric(label="Prediksi (USD)", value=date_pred.iloc[-1]["Prediksi"])

            time.sleep(287)
            
            data_aktual = get_binance_data(api_key, api_secret, selected_interval)
            harga_aktual = data_aktual['close'].iloc[-1]

            selisih_harga = prediction - harga_aktual

            # Menambahkan data baru ke dataframe
            aktual_selisih.loc[len(aktual_selisih)] = {"Aktual": harga_aktual, "Selisih Harga": selisih_harga}

            result = pd.concat([date_pred, aktual_selisih], axis=1)

            with placeholder.container():
                # menampilkan dataframe
                st.dataframe(data=result.iloc[-5:], use_container_width=True, hide_index=True)

                # menampilkan chart
                fig = px.line(result.iloc[-20:], x="Datetime", y=["Prediksi", "Aktual"], title="Chart Harga")
                fig.update_yaxes(title_text='Harga (USD)')
                st.write(fig)

        elif selected_interval == '15 menit' and now.second == 10 and now.minute % 15 == 0:
            current_time = now.strftime("%Y-%m-%d %H:%M")
            # Memanggil get_binance_data dengan interval yang sesuai
            data = get_binance_data(api_key, api_secret, selected_interval)
            # prediksi harga dari data pada fungsi get_data
            prediction = predict_price(data, selected_interval)

            # menambahkan data baru
            date_pred.loc[len(date_pred)] = {"Datetime": current_time, "Prediksi": prediction}

            with header.container():
                kpi1, kpi2 = st.columns(2)
                kpi1.metric(label="Datetime", value=date_pred.iloc[-1]["Datetime"])
                kpi2.metric(label="Prediksi (USD)", value=date_pred.iloc[-1]["Prediksi"])

            time.sleep(887)
            
            data_aktual = get_binance_data(api_key, api_secret, selected_interval)
            harga_aktual = data_aktual['close'].iloc[-1]

            selisih_harga = prediction - harga_aktual

            # Menambahkan data baru ke dataframe
            aktual_selisih.loc[len(aktual_selisih)] = {"Aktual": harga_aktual, "Selisih Harga": selisih_harga}

            result = pd.concat([date_pred, aktual_selisih], axis=1)

            with placeholder.container():
                # menampilkan dataframe
                st.dataframe(data=result.iloc[-5:], use_container_width=True, hide_index=True)

                # menampilkan chart
                fig = px.line(result.iloc[-20:], x="Datetime", y=["Prediksi", "Aktual"], title="Chart Harga")
                fig.update_yaxes(title_text='Harga (USD)')
                st.write(fig)

        elif selected_interval == '-':
            break


# kerangka Streamlit
st.set_page_config(page_title="Prediksi BTC/USD", page_icon="bitcoin.png")

hide_st_style = """
            <style>
            footer {visibility: hidden;}
            <style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Judul aplikasi Streamlit
st.title('Prediksi Harga Bitcoin(USD)')

selected_interval = st.selectbox('Pilih Timeframe: ', ['-', '1 menit', '5 menit', '15 menit'])

if selected_interval == '-':
    st.info("Silahkan memilih timeframe.")

# menjalankan fungsi scheduler
schedule = scheduler(selected_interval)