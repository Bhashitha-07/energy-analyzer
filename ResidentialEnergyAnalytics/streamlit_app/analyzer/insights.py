import pandas as pd

def analyze_usage_patterns(df):
    insights = []

    # Add an 'hour' column from the timestamp
    df['hour'] = df['timestamp'].dt.hour

    # Group by hour and calculate average usage
    hourly_avg = df.groupby('hour')['energy_usage'].mean()

    # Find the hour with the maximum average usage
    peak_hour = hourly_avg.idxmax()
    peak_value = hourly_avg.max()

    # Detect if usage is consistent or highly variable
    std_dev = hourly_avg.std()
    if std_dev > 10:
        variability = "highly variable"
    else:
        variability = "fairly consistent"

    insights.append(f"⏰ Peak energy usage is at {peak_hour}:00 hrs with an average of {peak_value:.2f} kWh.")
    insights.append(f"📊 Your hourly usage pattern is {variability}.")

    return insights, hourly_avg
