import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_hourly_cycle_usage(
    journeys_file="journeys_with_coordinates.csv",
    date_column="Start date",
    year=2025,
    output_name="hourly_santander_usage",
    show_title=False
):
    """
    Plot the average number of Santander Cycles journeys starting
    in each hour of the day, with one line for each weekday.

    Averages are calculated across all occurrences of each
    weekday in the selected year, including zero-journey days.

    Returns
    -------
    fig, ax, hourly_averages
    """

    # ==================================================
    # 1. LOAD DATA
    # ==================================================

    df = pd.read_csv(journeys_file)

    if date_column not in df.columns:
        raise ValueError(
            f"Column '{date_column}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # ==================================================
    # 2. PROCESS DATES AND TIMES
    # ==================================================

    df[date_column] = pd.to_datetime(
        df[date_column],
        dayfirst=True,
        errors="coerce"
    )

    df = df[
        df[date_column].dt.year == year
    ].copy()

    if df.empty:
        raise ValueError(f"No journeys found for {year}.")

    # Extract date and hour
    df["date"] = df[date_column].dt.normalize()
    df["hour"] = df[date_column].dt.hour

    # ==================================================
    # 3. COUNT JOURNEYS BY DATE AND HOUR
    # ==================================================

    hourly_counts = (
        df.groupby(["date", "hour"])
        .size()
        .unstack(fill_value=0)
    )

    # Ensure every hour is represented
    hourly_counts = hourly_counts.reindex(
        columns=range(24),
        fill_value=0
    )

    # Include all dates in the year, even those with
    # no recorded journeys
    all_dates = pd.date_range(
        start=f"{year}-01-01",
        end=f"{year}-12-31",
        freq="D"
    )

    hourly_counts = hourly_counts.reindex(
        all_dates,
        fill_value=0
    )

    # ==================================================
    # 4. CALCULATE AVERAGE BY WEEKDAY
    # ==================================================

    weekday_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    hourly_averages = (
        hourly_counts
        .groupby(hourly_counts.index.dayofweek)
        .mean()
        .reindex(range(7))
    )

    hourly_averages.index = weekday_names

    # ==================================================
    # 5. CREATE FIGURE
    # ==================================================

    fig, ax = plt.subplots(
        figsize=(11.69, 8.27)
    )

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    fig.subplots_adjust(
        left=0.11,
        right=0.96,
        top=0.91,
        bottom=0.13
    )

    # ==================================================
    # 6. PLOT SEVEN LINES
    # ==================================================

    colours = {
        "Monday": "#1F77B4",
        "Tuesday": "#FF7F0E",
        "Wednesday": "#2CA02C",
        "Thursday": "#D62728",
        "Friday": "#9467BD",
        "Saturday": "#8C564B",
        "Sunday": "#17BECF"
    }

    hours = np.arange(24)

    for day in weekday_names:

        ax.plot(
            hours,
            hourly_averages.loc[day].values,
            label=day,
            color=colours[day],
            linewidth=2.2,
            zorder=3
        )

    # ==================================================
    # 7. FORMAT AXES
    # ==================================================

    ax.set_xlim(0, 23)

    ax.set_xticks(np.arange(0, 24, 2))

    ax.set_xticklabels([
        f"{h:02d}:00"
        for h in range(0, 24, 2)
    ])

    ax.set_xlabel(
        "Time of day",
        fontsize=12,
        labelpad=12
    )

    ax.set_ylabel(
        "Average journeys starting per hour",
        fontsize=12,
        labelpad=12
    )

    ax.yaxis.set_major_formatter(
        ticker.StrMethodFormatter("{x:,.0f}")
    )

    ax.set_ylim(
        0,
        hourly_averages.to_numpy().max() * 1.12
    )

    ax.tick_params(
        axis="both",
        labelsize=11,
        length=0,
        pad=8
    )

    # ==================================================
    # 8. GRIDLINES AND BORDERS
    # ==================================================

    ax.grid(
        axis="y",
        color="#E6E6E6",
        linewidth=0.8,
        zorder=0
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.spines["bottom"].set_color("#CCCCCC")

    # ==================================================
    # 9. LEGEND
    # ==================================================

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=7,
        frameon=False,
        fontsize=9
    )

    # ==================================================
    # 10. OPTIONAL TITLE
    # ==================================================

    if show_title:
        ax.set_title(
            f"Average Hourly Santander Cycles Usage, {year}",
            fontsize=16,
            fontweight="bold",
            color="#222222",
            pad=40
        )

    # ==================================================
    # 11. EXPORT
    # ==================================================

    fig.savefig(
        f"{output_name}.png",
        dpi=300,
        facecolor="white"
    )

    fig.savefig(
        f"{output_name}.pdf",
        facecolor="white"
    )

    plt.show()

    print("\nAverage hourly journeys by weekday:")
    print(hourly_averages.round(1))

    return fig, ax, hourly_averages


fig, ax, hourly_averages = plot_hourly_cycle_usage(
    journeys_file="TestData.csv",
    date_column="Start date",
    year=2025,
    output_name="hourly_santander_usage",
    show_title=False
)