def generate_recommendations(df):
    recommendations = []

    # 1. High average usage
    avg_usage = df['energy_usage'].mean()
    if avg_usage > 25:
        recommendations.append("⚡ Your average energy use is high. Try reducing heavy appliance use during the day.")

    # 2. Peak usage spike
    peak_usage = df['energy_usage'].max()
    if peak_usage > avg_usage * 1.5:
        recommendations.append("📈 High energy spike detected. Consider running devices during off-peak hours.")

    # 3. Night-time usage
    if df['timestamp'].dt.hour.between(0, 6).sum() > 0:
        recommendations.append("🌙 Devices are active at night. Turn off idle appliances overnight to save energy.")

    return recommendations
