import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("TestData.csv")


def hourly_average_duration_chart(
    df,
    date_column="Start date",
    duration_column="duration_minutes"
):
    """
    Plot average journey duration for each hour of the day.
    """

    # Work with a copy
    data = df[[date_column, duration_column]].copy()

    # Convert start dates to datetime
    data[date_column] = pd.to_datetime(
        data[date_column],
        dayfirst=True,
        errors="coerce"
    )

    # Convert durations to numeric values
    data[duration_column] = pd.to_numeric(
        data[duration_column],
        errors="coerce"
    )

    # Remove missing or invalid values
    data = data.dropna(
        subset=[date_column, duration_column]
    )

    data = data[
        np.isfinite(data[duration_column])
        & (data[duration_column] > 0)
    ].copy()

    # Extract hour of journey start
    data["hour"] = data[date_column].dt.hour

    # Calculate mean journey duration for each hour
    hourly_durations = (
        data.groupby("hour")[duration_column]
        .mean()
        .reindex(range(24))
    )

    # Plot
    plt.figure(figsize=(12, 6))

    plt.bar(
        hourly_durations.index,
        hourly_durations.values,
        width=0.85
    )

    plt.xlabel("Hour of Day")
    plt.ylabel("Average Journey Duration (minutes)")
    #plt.title("Average Journey Duration by Hour of Day")

    plt.xticks(range(24))
    plt.xlim(-0.5, 23.5)

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    return hourly_durations


# Generate chart
hourly_duration_data = hourly_average_duration_chart(df)

# Print hourly averages
print(hourly_duration_data.round(2))