from recommender.tips import generate_recommendations
from analyzer.insights import analyze_usage_patterns
from chatbot.chat_bot import get_ai_response  # AI chatbot import

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import seaborn as sns

from sklearn.linear_model import LinearRegression

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("assets/style.css")

st.set_page_config(page_title="Energy Forecast", layout="wide")
st.title("🏡 Residential Energy Analytics + Forecasting")

# Upload CSV section
st.markdown("## 📤 Upload Your Energy Data CSV")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])

    st.subheader("📊 Raw Data")
    st.dataframe(df.head())

    df['rolling_avg'] = df['energy_usage'].rolling(window=24).mean()

    st.subheader("📈 Energy Usage Over Time")
    fig, ax = plt.subplots()
    ax.plot(df['timestamp'], df['energy_usage'], color='green')
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy Usage (kWh)")
    st.pyplot(fig)

    st.subheader("💡 Smart Energy Recommendations")
    tips = generate_recommendations(df)
    if tips:
        for tip in tips:
            st.success(tip)
    else:
        st.info("✅ Your energy usage looks efficient!")

    st.subheader("📌 Energy Usage Analyzer")
    insights, hourly_avg = analyze_usage_patterns(df)
    for insight in insights:
        st.info(insight)

    st.subheader("📉 Average Energy Usage by Hour")
    fig2, ax2 = plt.subplots()
    ax2.bar(hourly_avg.index, hourly_avg.values, color='orange')
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Avg Usage (kWh)")
    st.pyplot(fig2)

    st.subheader("🕒 Hour vs Day Energy Usage Heatmap")
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['hour'] = df['timestamp'].dt.hour
    pivot_table = df.pivot_table(index='day_of_week', columns='hour', values='energy_usage', aggfunc='mean')
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot_table, cmap="YlGnBu", ax=ax3)
    st.pyplot(fig3)

    st.subheader("🚨 Anomaly Detection in Energy Usage")
    df['z_score'] = (df['energy_usage'] - df['energy_usage'].mean()) / df['energy_usage'].std()
    df['anomaly'] = df['z_score'].apply(lambda x: abs(x) > 2.5)
    fig4, ax4 = plt.subplots()
    ax4.plot(df['timestamp'], df['energy_usage'], label='Energy Usage', color='blue')
    ax4.scatter(df[df['anomaly']]['timestamp'], df[df['anomaly']]['energy_usage'], color='red', label='Anomalies')
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Energy Usage (kWh)")
    ax4.legend()
    st.pyplot(fig4)

    st.subheader("🧠 Optimized Energy Usage Suggestions")
    top_usage = df.groupby(df['hour'])['energy_usage'].mean().sort_values(ascending=False).head(3)
    for hour, usage in top_usage.items():
        st.warning(f"⚠️ Hour: **{hour}:00 - {hour+1}:00** → Avg Usage: {usage:.2f} kWh")
    st.markdown("✅ **Suggestions to Optimize Usage:**")
    st.info("• Run heavy appliances during off-peak hours.")
    st.info("• Use smart timers to automate energy-intensive tasks.")
    st.info("• Consider solar if your peak is during the day.")

    st.subheader("🔮 Forecast: Next 7 Days Energy Usage")
    df = df.dropna(subset=['timestamp', 'energy_usage'])
    df['timestamp_ordinal'] = df['timestamp'].map(datetime.datetime.toordinal)
    df_sorted = df.sort_values("timestamp")

    if not df_sorted.empty:
        X = df_sorted[['timestamp_ordinal']]
        y = df_sorted['energy_usage']
        model = LinearRegression()
        model.fit(X, y)

        last_date = df_sorted['timestamp'].max()
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 8)]
        future_dates_ord = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
        future_preds = model.predict(future_dates_ord)

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Energy Usage (kWh)": future_preds
        })
        st.dataframe(forecast_df)

        fig5, ax5 = plt.subplots()
        ax5.plot(df_sorted['timestamp'], df_sorted['energy_usage'], label="Historical", color='blue')
        ax5.plot(forecast_df['Date'], forecast_df["Predicted Energy Usage (kWh)"], label="Forecast", color='orange')
        ax5.set_xlabel("Date")
        ax5.set_ylabel("Energy Usage (kWh)")
        ax5.legend()
        st.pyplot(fig5)
    else:
        st.warning("⚠️ Not enough data to forecast.")

    st.subheader("🔁 24-Hour Rolling Average Energy Usage")
    fig6, ax6 = plt.subplots()
    ax6.plot(df['timestamp'], df['energy_usage'], label='Original', color='gray', alpha=0.4)
    ax6.plot(df['timestamp'], df['rolling_avg'], label='24-Hour Avg', color='purple')
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Energy Usage (kWh)")
    ax6.legend()
    st.pyplot(fig6)
else:
    st.info("📁 Please upload a CSV file to begin.")

# ---------------------------
# 💬 AI Chatbot Section
# ---------------------------
st.markdown("---")
st.header("💬 AI Chat Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask me anything about your energy usage:")

if user_input:
    response = get_ai_response(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**🧑 {speaker}:** {message}")
    else:
        st.markdown(f"**🤖 {speaker}:** {message}")
