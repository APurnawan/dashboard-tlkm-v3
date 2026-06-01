import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#LOAD DATA
st.set_page_config(
    page_title="TLKM Stock Forecast Dashboard",
    layout="wide"
)

@st.cache_data(ttl=3600)
def load_stock(period):
    try:
        df = yf.download(
            "TLKM.JK",
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False
        )

        return df

    except Exception as e:
        st.error(f"Error download data: {e}")
        return pd.DataFrame()

forecast_df = pd.read_csv("forecast_30.csv")
actual_pred_df = pd.read_csv("actual_vs_prediction.csv")
loss_df = pd.read_csv("loss_history.csv")
metrics_df = pd.read_csv("metrics.csv")

forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])

#SIDE BAR
st.sidebar.title("Kontrol Dashboard")

periode = st.sidebar.selectbox(
    "Periode Historis",
    ["1y","3y","5y","10y"],
    index=3
)

comparison = st.sidebar.multiselect(
    "Saham Pembanding",
    ["BBCA.JK","BBRI.JK","BMRI.JK","ASII.JK"],
    default=["BBCA.JK","BBRI.JK","BMRI.JK","ASII.JK"]
)

df = load_stock(periode)

#INDIKATOR
df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()

delta = df["Close"].diff()

gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

df["RSI"] = 100 - (100 / (1 + rs))

ema12 = df["Close"].ewm(span=12).mean()
ema26 = df["Close"].ewm(span=26).mean()

df["MACD"] = ema12 - ema26
df["Signal"] = df["MACD"].ewm(span=9).mean()

#KPI ATAS
last_close = df["Close"].iloc[-1].item()
prev_close = df["Close"].iloc[-2].item()

change = last_close - prev_close
change_pct = (change / prev_close) * 100

volume = int(df["Volume"].iloc[-1].item())

signal = "BUY" if df["MA20"].iloc[-1] > df["MA50"].iloc[-1] else "SELL"

st.title("TLKM Stock Forecast Dashboard")

c1,c2,c3,c4,c5 = st.columns(5)

c1.metric(
    "Harga Saat Ini",
    f"Rp {last_close:,.0f}"
)

c2.metric(
    "Perubahan Harian",
    f"Rp {change:,.0f}",
    f"{change_pct:.2f}%"
)

c3.metric(
    "Volume",
    f"{volume:,.0f}"
)

c4.metric(
    "Sinyal",
    signal
)

c5.metric(
    "Model",
    "Multivariate LSTM"
)

#TAB 1
tab1,tab2,tab3,tab4 = st.tabs([
    "Historis & Forecast",
    "Evaluasi Model",
    "Perbandingan Saham",
    "Data Mentah"
])

#TAB1 HISTORIS
with tab1:

    left,right = st.columns([3,1])

#GRAPHIC HISTORIS
with left:

    st.subheader("Grafik Historis TLKM")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Close Price"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MA20"],
            name="MA20"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MA50"],
            name="MA50"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

#FORECAST
    st.subheader(
        "Forecast 30 Hari Menggunakan Multivariate LSTM"
    )

    actual_part = df.tail(90)
    
    last_actual = pd.DataFrame({
        "Date":[actual_part.index[-1]],
        "Forecast":[actual_part["Close"].iloc[-1]]
    })

    forecast_plot = pd.concat([
        last_actual,
        forecast_df
    ])

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=actual_part.index,
            y=actual_part["Close"],
            name="Historical Close"
        )
    )

    fig2.add_trace(
    go.Scatter(
        x=forecast_plot["Date"],
        y=forecast_plot["Forecast"],
        name="Forecast LSTM",
        line=dict(dash="dash")
         )
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

#PREDIKSI

    p1,p2,p3 = st.columns(3)

    p1.metric(
        "Prediksi Besok",
        f"Rp {forecast_df['Forecast'].iloc[0]:,.0f}"
    )

    p2.metric(
        "Prediksi 7 Hari",
        f"Rp {forecast_df['Forecast'].iloc[6]:,.0f}"
    )

    p3.metric(
        "Prediksi 30 Hari",
        f"Rp {forecast_df['Forecast'].iloc[-1]:,.0f}"
    )

#TABEL FORECAST
    st.subheader("Tabel Forecast 30 Hari")

    st.dataframe(
        forecast_df,
        use_container_width=True
    )

    st.subheader("Volume Perdagangan")

    vol_fig = go.Figure()

    vol_fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume"
        )
    )

st.plotly_chart(
    vol_fig,
    use_container_width=True
)
#PANEL KANAN
with right:

    st.subheader("Ringkasan Harga")

    st.write(
        f"Open : Rp {float(df['Open'].iloc[-1]):,.0f}"
    )

    st.write(
        f"High : Rp {float(df['High'].iloc[-1]):,.0f}"
    )

    st.write(
        f"Low : Rp {float(df['Low'].iloc[-1]):,.0f}"
    )

    st.write(
        f"Close : Rp {float(df['Close'].iloc[-1]):,.0f}"
    )

    st.write(
        f"52W High : Rp {float(df['Close'].tail(252).max()):,.0f}"
    )

    st.write(
        f"52W Low : Rp {float(df['Close'].tail(252).min()):,.0f}"
    )

#STATISTIK
    st.subheader("Ringkasan Statistik")

    stat_df = pd.DataFrame({
        "Statistik":[
            "Harga Tertinggi",
            "Harga Terendah",
            "Rata-rata Close",
            "Standar Deviasi"
        ],
        "Nilai":[
            df["Close"].max(),
            df["Close"].min(),
            df["Close"].mean(),
            df["Close"].std()
        ]
    })

    st.dataframe(
        stat_df,
        use_container_width=True
    )

#RSI MACD
    st.subheader("Indikator Teknikal")

    st.write(
        f"RSI : {df['RSI'].iloc[-1]:.2f}"
    )

    st.write(
        f"MACD : {df['MACD'].iloc[-1]:.2f}"
    )

    st.write(
        f"Signal : {df['Signal'].iloc[-1]:.2f}"
    )

    st.write(
        f"MA20 : {df['MA20'].iloc[-1]:.2f}"
    )

    st.write(
        f"MA50 : {df['MA50'].iloc[-1]:.2f}"
    )

#TAB 2 EVALUASI
with tab2:

    st.subheader("Informasi Model")

    info = pd.DataFrame({
        "Parameter":[
            "Model",
            "Lookback",
            "Epoch",
            "Jumlah Fitur"
        ],
        "Value":[
            "Multivariate LSTM",
            "60",
            "50",
            "8"
        ]
    })

    st.dataframe(info)

#MATRIX
    cols = st.columns(len(metrics_df))

    for i,row in metrics_df.iterrows():

        cols[i].metric(
            row["Metric"],
            f"{row['Value']:.4f}"
        )

#ACTUAL VS PREDIKSI
    fig3 = go.Figure()

    fig3.add_trace(
        go.Scatter(
            y=actual_pred_df["Actual"],
            name="Actual"
        )
    )

    fig3.add_trace(
        go.Scatter(
            y=actual_pred_df["Prediction"],
            name="Prediction"
        )
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.subheader("Tabel Actual vs Prediction")

    st.dataframe(
        actual_pred_df,
        use_container_width=True
    )
    

#LOSS CURVE
    fig4 = go.Figure()

    fig4.add_trace(
        go.Scatter(
            y=loss_df["loss"],
            name="Training Loss"
        )
    )

    fig4.add_trace(
        go.Scatter(
            y=loss_df["val_loss"],
            name="Validation Loss"
        )
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    st.subheader("Tabel History Loss")

st.dataframe(
    loss_df,
    use_container_width=True
)

#TAB 3 PERBANDINGAN
with tab3:

    compare_fig = go.Figure()

    compare_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="TLKM"
        )
    )

    for stock in comparison:

        tmp = yf.download(
            stock,
            period=periode,
            auto_adjust=True,
            progress=False
        )

        compare_fig.add_trace(
            go.Scatter(
                x=tmp.index,
                y=tmp["Close"],
                name=stock.replace(".JK","")
            )
        )

    st.plotly_chart(
        compare_fig,
        use_container_width=True
    )

#TAB 4 DATA MENTAH
with tab4:

    st.subheader(
        "Data Mentah Historis TLKM"
    )

    st.dataframe(
        df.reset_index(),
        use_container_width=True
    )                    

st.caption(
    "Data historis diperoleh dari Yahoo Finance. Forecast menggunakan model Multivariate LSTM."
)
