import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_marathon_active_rentals(
    df,
    comparison_date="2025-04-06",
    marathon_date="2025-04-27",
    date_column="Start date",
    duration_column="duration_minutes",
    interval_minutes=5,
    output_name="marathon_active_rentals"
):
    """
    Compare the number of Santander Cycles bicycles actively rented
    throughout two dates.

    Includes:
        - Active rentals throughout both days
        - Peak simultaneous rentals
        - Mean journey duration
        - Percentage of journeys exceeding 30 minutes

    Active rentals are calculated from journey start and end times.
    Journeys beginning before midnight but continuing into the
    selected date are included.

    Saves:
        output_name.pdf
        output_name.png

    Returns:
        Dictionary containing time series and summary statistics.
    """

    data = df.copy()

    # --------------------------------------------------
    # Prepare dates and durations
    # --------------------------------------------------

    data[date_column] = pd.to_datetime(
        data[date_column],
        dayfirst=True,
        errors="coerce"
    )

    if duration_column in data.columns:
        data["_duration_min"] = pd.to_numeric(
            data[duration_column],
            errors="coerce"
        )

    elif "Total duration (ms)" in data.columns:
        data["_duration_min"] = (
            pd.to_numeric(
                data["Total duration (ms)"],
                errors="coerce"
            ) / 60000
        )

    else:
        raise ValueError("No journey duration column found.")

    data = data.dropna(
        subset=[date_column, "_duration_min"]
    )

    data = data[data["_duration_min"] > 0].copy()

    data["_end_date"] = (
        data[date_column]
        + pd.to_timedelta(
            data["_duration_min"],
            unit="m"
        )
    )

    comparison_date = pd.Timestamp(comparison_date).normalize()
    marathon_date = pd.Timestamp(marathon_date).normalize()

    # --------------------------------------------------
    # Calculate active rentals
    # --------------------------------------------------

    def calculate_active_rentals(day):

        start = day
        end = day + pd.Timedelta(days=1)

        # Include journeys overlapping the selected day
        relevant = data[
            (data[date_column] < end)
            & (data["_end_date"] > start)
        ]

        times = pd.date_range(
            start=start,
            end=end,
            freq=f"{interval_minutes}min",
            inclusive="left"
        )

        # Count rentals active at each time
        starts = np.sort(
            relevant[date_column].values.astype("datetime64[ns]")
        )

        ends = np.sort(
            relevant["_end_date"].values.astype("datetime64[ns]")
        )

        active = (
            np.searchsorted(
                starts,
                times.values,
                side="right"
            )
            -
            np.searchsorted(
                ends,
                times.values,
                side="right"
            )
        )

        return pd.Series(
            active,
            index=times
        )

    comparison_active = calculate_active_rentals(
        comparison_date
    )

    marathon_active = calculate_active_rentals(
        marathon_date
    )

    # --------------------------------------------------
    # Journey duration statistics
    # --------------------------------------------------

    def duration_stats(day):

        journeys = data[
            data[date_column].dt.normalize() == day
        ]

        durations = journeys["_duration_min"]

        return {
            "total_journeys": len(journeys),
            "mean_duration": durations.mean(),
            "median_duration": durations.median(),
            "over_30_percent": (
                (durations > 30).mean() * 100
            )
        }

    stats_comparison = duration_stats(comparison_date)
    stats_marathon = duration_stats(marathon_date)

    # --------------------------------------------------
    # Peak rental statistics
    # --------------------------------------------------

    peak_comparison = comparison_active.max()
    peak_marathon = marathon_active.max()

    peak_change = (
        (peak_marathon - peak_comparison)
        / peak_comparison
        * 100
    )

    journey_change = (
        (
            stats_marathon["total_journeys"]
            - stats_comparison["total_journeys"]
        )
        / stats_comparison["total_journeys"]
        * 100
    )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 5.5))

    blue = "#2874A6"
    red = "#C0392B"

    hours = (
        comparison_active.index.hour
        + comparison_active.index.minute / 60
    )

    # Comparison Sunday
    ax.plot(
        hours,
        comparison_active.values,
        color=blue,
        linewidth=2.2,
        label=(
            f"6 April "
            f"(peak: {peak_comparison:,})"
        )
    )

    # Marathon Day
    ax.plot(
        hours,
        marathon_active.values,
        color=red,
        linewidth=2.2,
        label=(
            f"27 April — Marathon Day "
            f"(peak: {peak_marathon:,})"
        )
    )

    # --------------------------------------------------
    # Annotations
    # --------------------------------------------------

    ax.text(
        0.02,
        0.96,
        f"Peak active rentals: {peak_change:+.1f}%",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top"
    )

    ax.text(
        0.02,
        0.89,
        f"Total journeys: {journey_change:+.1f}%",
        transform=ax.transAxes,
        fontsize=10,
        va="top"
    )

    # Duration summary box
    summary_text = (
        "Journey characteristics\n"
        "\n"
        "Mean duration\n"
        f"6 Apr: {stats_comparison['mean_duration']:.2f} min\n"
        f"27 Apr: {stats_marathon['mean_duration']:.2f} min\n"
        "\n"
        "Journeys over 30 min\n"
        f"6 Apr: {stats_comparison['over_30_percent']:.2f}%\n"
        f"27 Apr: {stats_marathon['over_30_percent']:.2f}%"
    )

    ax.text(
        0.98,
        0.96,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="right",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            edgecolor="#DDDDDD",
            alpha=0.95
        )
    )

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------

    ax.set_xlabel(
        "Hour of day",
        fontsize=11
    )

    ax.set_ylabel(
        "Number of bicycles actively rented",
        fontsize=11
    )

    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlim(0, 24)

    ax.set_ylim(
        0,
        max(
            peak_comparison,
            peak_marathon
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
        ncol=2,
        frameon=False,
        fontsize=10
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

    print("\nACTIVE RENTAL COMPARISON")
    print("-" * 50)

    print(f"6 April peak:       {peak_comparison:,}")
    print(f"27 April peak:      {peak_marathon:,}")
    print(f"Peak change:        {peak_change:+.2f}%")

    print(f"\nTotal journeys change: {journey_change:+.2f}%")

    print("\nMEAN JOURNEY DURATION")
    print(
        f"6 April:  "
        f"{stats_comparison['mean_duration']:.2f} min"
    )
    print(
        f"27 April: "
        f"{stats_marathon['mean_duration']:.2f} min"
    )

    print("\nJOURNEYS EXCEEDING 30 MINUTES")
    print(
        f"6 April:  "
        f"{stats_comparison['over_30_percent']:.2f}%"
    )
    print(
        f"27 April: "
        f"{stats_marathon['over_30_percent']:.2f}%"
    )

    return {
        "comparison_active": comparison_active,
        "marathon_active": marathon_active,
        "comparison_stats": stats_comparison,
        "marathon_stats": stats_marathon,
        "peak_change_percent": peak_change
    }


# --------------------------------------------------
# Run
# --------------------------------------------------

df = pd.read_csv("TestData.csv")

results = plot_marathon_active_rentals(
    df,
    comparison_date="2025-04-06",
    marathon_date="2025-04-27",
    interval_minutes=5
)