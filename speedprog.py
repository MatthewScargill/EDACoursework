import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def hourly_speed_by_weekday(
    df,
    date_column="Start date",
    speed_column="average_speed_kmh"
):
    """
    Plot average implied journey speed by hour of day,
    with a separate line for each day of the week.
    """

    # Work with a copy of the dataset
    data = df[[date_column, speed_column]].copy()

    # Convert journey start times to datetime
    data[date_column] = pd.to_datetime(
        data[date_column],
        dayfirst=True,
        errors="coerce"
    )

    # Convert speeds to numeric values
    data[speed_column] = pd.to_numeric(
        data[speed_column],
        errors="coerce"
    )

    # Remove missing or non-finite values
    data = data.dropna(subset=[date_column, speed_column])

    data = data[
        np.isfinite(data[speed_column])
        & (data[speed_column] >= 0)
    ].copy()

    # Extract hour and day of week
    data["hour"] = data[date_column].dt.hour
    data["weekday"] = data[date_column].dt.dayofweek

    # Calculate mean implied speed for each weekday and hour
    hourly_speeds = (
        data.groupby(["hour", "weekday"])[speed_column]
        .mean()
        .unstack("weekday")
        .reindex(index=range(24), columns=range(7))
    )

    # Weekday labels
    weekday_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for day_number, day_name in enumerate(weekday_names):
        ax.plot(
            hourly_speeds.index,
            hourly_speeds[day_number],
            label=day_name,
            linewidth=2,
            zorder=3
        )

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Implied Journey Speed (km/h)")

    ax.set_xticks(range(24))
    ax.set_xlim(0, 23)

    # Horizontal gridlines only, behind the lines
    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        linestyle="-",
        alpha=0.3
    )

    # Remove surrounding box
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove tick marks while keeping labels
    ax.tick_params(
        axis="both",
        length=0
    )

    # Keep legend without a surrounding box
    ax.legend(
        title="Day of Week",
        frameon=False
    )

    plt.tight_layout()
    plt.show()

    return hourly_speeds


# Load cleaned dataset
df = pd.read_csv("TestData.csv")

# Generate chart
hourly_speed_data = hourly_speed_by_weekday(df)