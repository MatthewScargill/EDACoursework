import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_weekday_cycle_usage(
    journeys_file="journeys_with_coordinates.csv",
    date_column="Start date",
    year=2025,
    output_name="weekday_santander_usage",
    show_title=False
):
    """
    Plot average daily Santander Cycles journeys by day of the week
    for three periods:
        1. January and February
        2. June and July
        3. Entire year

    Average = total journeys on a given weekday /
              number of occurrences of that weekday in the period.

    Returns
    -------
    fig, ax, results
        Matplotlib figure, axes, and DataFrame of average journeys.
    """

    # ==================================================
    # 1. LOAD DATA
    # ==================================================

    df = pd.read_csv(journeys_file)

    if date_column not in df.columns:
        raise ValueError(
            f"Column '{date_column}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    # ==================================================
    # 2. PROCESS DATES
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

    df["date"] = df[date_column].dt.normalize()

    # ==================================================
    # 3. DEFINE PERIODS
    # ==================================================

    periods = {
        "January–February": [1, 2],
        "June–July": [6, 7],
        "Entire year": list(range(1, 13))
    }

    weekdays = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    results = pd.DataFrame(index=weekdays)

    # ==================================================
    # 4. CALCULATE AVERAGE JOURNEYS PER WEEKDAY
    # ==================================================

    for period_name, months in periods.items():

        period_df = df[
            df[date_column].dt.month.isin(months)
        ]

        # Count journeys occurring on each date
        daily_counts = period_df.groupby("date").size()

        # Create a complete calendar for the period
        # This includes days with zero recorded journeys
        calendar_dates = pd.date_range(
            start=f"{year}-01-01",
            end=f"{year}-12-31",
            freq="D"
        )

        calendar_dates = calendar_dates[
            calendar_dates.month.isin(months)
        ]

        # Include zero-journey days
        daily_counts = daily_counts.reindex(
            calendar_dates,
            fill_value=0
        )

        # Average across all occurrences of each weekday
        weekday_averages = (
            daily_counts
            .groupby(daily_counts.index.dayofweek)
            .mean()
            .reindex(range(7), fill_value=0)
        )

        results[period_name] = weekday_averages.values

    # ==================================================
    # 5. CREATE FIGURE
    # ==================================================

    fig, ax = plt.subplots(
        figsize=(11.69, 8.27)
    )

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    fig.subplots_adjust(
        left=0.12,
        right=0.96,
        top=0.92,
        bottom=0.14
    )

    # ==================================================
    # 6. GROUPED BAR CHART
    # ==================================================

    x = np.arange(len(weekdays))
    width = 0.25

    colours = {
        "January–February": "#7FA9C4",
        "June–July": "#C0392B",
        "Entire year": "#555555"
    }

    for i, period_name in enumerate(periods):

        offset = (i - 1) * width

        ax.bar(
            x + offset,
            results[period_name],
            width=width,
            label=period_name,
            color=colours[period_name],
            edgecolor="none",
            zorder=3
        )

    # ==================================================
    # 7. FORMAT AXES
    # ==================================================

    ax.set_xticks(x)

    ax.set_xticklabels(
        weekdays,
        fontsize=11
    )

    ax.set_xlabel(
        "Day of the week",
        fontsize=12,
        labelpad=12
    )

    ax.set_ylabel(
        "Average journeys per day",
        fontsize=12,
        labelpad=12
    )

    ax.yaxis.set_major_formatter(
        ticker.StrMethodFormatter("{x:,.0f}")
    )

    ax.set_ylim(
        0,
        results.to_numpy().max() * 1.15
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
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        frameon=False,
        fontsize=11
    )

    # ==================================================
    # 10. OPTIONAL TITLE
    # ==================================================

    if show_title:
        ax.set_title(
            f"Average Santander Cycles Journeys by Weekday, {year}",
            fontsize=16,
            fontweight="bold",
            color="#222222",
            pad=35
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

    # Display calculated values
    print("\nAverage journeys per day of the week:")
    print(results.round(1))

    return fig, ax, results


fig, ax, results = plot_weekday_cycle_usage(
    journeys_file="TestData.csv",
    date_column="Start date",
    year=2025,
    output_name="weekday_santander_usage",
    show_title=False
)