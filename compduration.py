import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_marathon_hourly_comparison(
    df,
    comparison_date="2025-04-06",
    marathon_date="2025-04-27",
    date_column="Start date",
    output_name="marathon_hourly_comparison",
    average_year=2025,
    exclude_marathon_from_average=True
):
    """
    Compare hourly Santander Cycles journey departures for:

        1. Comparison Sunday
        2. Marathon Day
        3. Average Sunday in the specified year

    The average Sunday is calculated using all Sundays in the
    specified year, optionally excluding Marathon Day.

    Saves:
        output_name.pdf
        output_name.png

    Returns:
        DataFrame containing hourly journey counts.
    """

    data = df.copy()

    # --------------------------------------------------
    # Prepare dates
    # --------------------------------------------------

    data[date_column] = pd.to_datetime(
        data[date_column],
        dayfirst=True,
        errors="coerce"
    )

    data = data.dropna(subset=[date_column])

    data["_date"] = data[date_column].dt.normalize()
    data["_hour"] = data[date_column].dt.hour

    comparison_date = pd.Timestamp(comparison_date).normalize()
    marathon_date = pd.Timestamp(marathon_date).normalize()

    comparison = data[data["_date"] == comparison_date]
    marathon = data[data["_date"] == marathon_date]

    if comparison.empty or marathon.empty:
        raise ValueError(
            "No journeys found for one or both selected dates."
        )

    # --------------------------------------------------
    # Hourly counts for selected dates
    # --------------------------------------------------

    hours = np.arange(24)

    comparison_hourly = (
        comparison["_hour"]
        .value_counts()
        .reindex(hours, fill_value=0)
        .sort_index()
    )

    marathon_hourly = (
        marathon["_hour"]
        .value_counts()
        .reindex(hours, fill_value=0)
        .sort_index()
    )

    # --------------------------------------------------
    # Calculate average Sunday
    # --------------------------------------------------

    sundays = data[
        (data[date_column].dt.year == average_year)
        & (data[date_column].dt.dayofweek == 6)
    ].copy()

    if exclude_marathon_from_average:
        sundays = sundays[
            sundays["_date"] != marathon_date
        ]

    if sundays.empty:
        raise ValueError(
            "No Sundays found for the selected average year."
        )

    # Count journeys for each Sunday and hour.
    # Missing hours are filled with zero.

    sunday_hourly = (
        sundays.groupby(["_date", "_hour"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=hours, fill_value=0)
    )

    average_sunday_hourly = sunday_hourly.mean(axis=0)

    number_of_sundays = len(sunday_hourly)

    # --------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------

    comparison_total = len(comparison)
    marathon_total = len(marathon)

    average_sunday_total = average_sunday_hourly.sum()

    overall_change = (
        (marathon_total - comparison_total)
        / comparison_total
        * 100
    )

    average_change = (
        (marathon_total - average_sunday_total)
        / average_sunday_total
        * 100
    )

    comparison_evening = comparison_hourly.loc[19:23].sum()
    marathon_evening = marathon_hourly.loc[19:23].sum()
    average_evening = average_sunday_hourly.loc[19:23].sum()

    evening_change = (
        (marathon_evening - comparison_evening)
        / comparison_evening
        * 100
    )

    evening_average_change = (
        (marathon_evening - average_evening)
        / average_evening
        * 100
    )

    comparison_daytime = comparison_hourly.loc[10:15].sum()
    marathon_daytime = marathon_hourly.loc[10:15].sum()
    average_daytime = average_sunday_hourly.loc[10:15].sum()

    daytime_change = (
        (marathon_daytime - comparison_daytime)
        / comparison_daytime
        * 100
    )

    daytime_average_change = (
        (marathon_daytime - average_daytime)
        / average_daytime
        * 100
    )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 5.5))

    red = "#C0392B"
    blue = "#2874A6"
    grey = "#555555"

    # Highlight evening period
    ax.axvspan(
        18.5,
        23.5,
        color="#F2F2F2",
        zorder=0
    )

    # Average Sunday
    ax.plot(
        hours,
        average_sunday_hourly.values,
        color=grey,
        linewidth=2.2,
        linestyle="--",
        label=(
            f"Average Sunday "
            f"({average_sunday_total:,.0f} journeys)"
        ),
        zorder=2
    )

    # Comparison Sunday
    ax.plot(
        hours,
        comparison_hourly.values,
        color=blue,
        linewidth=2.4,
        marker="o",
        markersize=4,
        label=(
            f"{comparison_date.strftime('%-d %B')} "
            f"({comparison_total:,} journeys)"
        ),
        zorder=3
    )

    # Marathon Day
    ax.plot(
        hours,
        marathon_hourly.values,
        color=red,
        linewidth=2.4,
        marker="o",
        markersize=4,
        label=(
            f"{marathon_date.strftime('%-d %B')} — Marathon Day "
            f"({marathon_total:,} journeys)"
        ),
        zorder=4
    )

    # --------------------------------------------------
    # Annotations
    # --------------------------------------------------

    ax.text(
        0.02,
        0.96,
        f"Marathon vs 6 April: {overall_change:+.1f}%",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left"
    )

    ax.text(
        0.02,
        0.89,
        f"Marathon vs average Sunday: {average_change:+.1f}%",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left"
    )

    ax.text(
        0.98,
        0.96,
        f"Evening vs 6 April: {evening_change:+.1f}%",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="right",
        color=red
    )

    ax.text(
        0.98,
        0.89,
        f"Evening vs average: {evening_average_change:+.1f}%",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="right",
        color=red
    )

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------

    ax.set_xlabel(
        "Hour of day",
        fontsize=11
    )

    ax.set_ylabel(
        "Number of journeys beginning",
        fontsize=11
    )

    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(0, 23)

    ax.set_ylim(
        0,
        max(
            comparison_hourly.max(),
            marathon_hourly.max(),
            average_sunday_hourly.max()
        ) * 1.18
    )

    ax.yaxis.set_major_locator(
        MaxNLocator(integer=True)
    )

    ax.grid(
        axis="y",
        color="#DDDDDD",
        linewidth=0.8,
        alpha=0.7
    )

    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        labelsize=10
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        fontsize=9
    )

    plt.tight_layout()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    plt.savefig(
        f"{output_name}.pdf",
        bbox_inches="tight"
    )

    plt.savefig(
        f"{output_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # --------------------------------------------------
    # Print statistics
    # --------------------------------------------------

    print("\nMARATHON DAY COMPARISON")
    print("-" * 55)

    print(f"Comparison Sunday: {comparison_total:,} journeys")
    print(f"Marathon Day:      {marathon_total:,} journeys")
    print(f"Average Sunday:    {average_sunday_total:,.0f} journeys")

    print(f"\nSundays included in average: {number_of_sundays}")

    print("\nMARATHON VS COMPARISON SUNDAY")
    print(f"Overall change:    {overall_change:+.2f}%")
    print(f"Daytime change:    {daytime_change:+.2f}%")
    print(f"Evening change:    {evening_change:+.2f}%")

    print("\nMARATHON VS AVERAGE SUNDAY")
    print(f"Overall change:    {average_change:+.2f}%")
    print(f"Daytime change:    {daytime_average_change:+.2f}%")
    print(f"Evening change:    {evening_average_change:+.2f}%")

    print("\nPEAK DEPARTURE HOURS")
    print(
        f"Comparison Sunday: "
        f"{comparison_hourly.idxmax():02d}:00"
    )
    print(
        f"Marathon Day:      "
        f"{marathon_hourly.idxmax():02d}:00"
    )
    print(
        f"Average Sunday:    "
        f"{average_sunday_hourly.idxmax():02d}:00"
    )

    # --------------------------------------------------
    # Return hourly data
    # --------------------------------------------------

    return pd.DataFrame({
        "Hour": hours,
        "Comparison Sunday": comparison_hourly.values,
        "Marathon Day": marathon_hourly.values,
        "Average Sunday": average_sunday_hourly.values,
        "Difference vs Comparison": (
            marathon_hourly.values
            - comparison_hourly.values
        ),
        "Difference vs Average": (
            marathon_hourly.values
            - average_sunday_hourly.values
        )
    })


# --------------------------------------------------
# Run
# --------------------------------------------------

df = pd.read_csv("TestData.csv")

hourly_results = plot_marathon_hourly_comparison(
    df,
    comparison_date="2025-04-06",
    marathon_date="2025-04-27",
    average_year=2025
)