import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_station_net_flow_scatter(
    input_file,
    output_file="station_net_flow_scatter.png",
    date_column="Start date",
    start_station_column="Start station",
    end_station_column="End station",
    duration_column="Total duration (ms)",
    duration_unit="ms",
    morning_hours=(7, 10),
    evening_hours=(16, 19),
    min_station_journeys=20,
    balance_threshold=0.5,
    label_top_n=8,
    log_linthresh=0.5,
    min_bubble_size=20,
    max_bubble_size=350,
    figsize=(11, 9),
    dpi=300,
    show_plot=True
):
    """
    Plot average weekday morning versus evening net flow
    for Santander Cycles docking stations.

    Uses a symmetric logarithmic scale (symlog) to spread
    out stations near the origin while preserving positive,
    negative and zero values.

    Net flow = arrivals - departures.

    X-axis: Average weekday morning net flow.
    Y-axis: Average weekday evening net flow.

    Bubble size represents average daily station activity.

    Arrival times are calculated from journey start times
    and durations.

    Returns
    -------
    pandas.DataFrame
        Station-level net-flow statistics.
    """

    # ---------------------------------------------------------
    # 1. LOAD AND VALIDATE DATA
    # ---------------------------------------------------------

    df = pd.read_csv(input_file)

    required_columns = [
        date_column,
        start_station_column,
        end_station_column,
        duration_column
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if duration_unit not in ["ms", "seconds", "minutes"]:
        raise ValueError(
            "duration_unit must be 'ms', 'seconds', or 'minutes'."
        )

    if log_linthresh <= 0:
        raise ValueError(
            "log_linthresh must be greater than zero."
        )

    for period in [morning_hours, evening_hours]:
        if not (0 <= period[0] < period[1] <= 24):
            raise ValueError("Invalid time period.")

    df[date_column] = pd.to_datetime(
        df[date_column],
        dayfirst=True,
        errors="coerce"
    )

    df[duration_column] = pd.to_numeric(
        df[duration_column],
        errors="coerce"
    )

    df = df.dropna(
        subset=required_columns
    ).copy()

    df = df[
        df[duration_column] > 0
    ].copy()

    if df.empty:
        raise ValueError("No valid journeys remain.")

    # ---------------------------------------------------------
    # 2. CALCULATE JOURNEY END TIMES
    # ---------------------------------------------------------

    duration_units = {
        "ms": "ms",
        "seconds": "s",
        "minutes": "m"
    }

    durations = pd.to_timedelta(
        df[duration_column],
        unit=duration_units[duration_unit]
    )

    df["journey_start"] = df[date_column]
    df["journey_end"] = df["journey_start"] + durations

    # ---------------------------------------------------------
    # 3. DEFINE OBSERVED WEEKDAYS
    # ---------------------------------------------------------

    first_date = df["journey_start"].min().normalize()
    last_date = df["journey_start"].max().normalize()

    observed_dates = pd.date_range(
        first_date,
        last_date,
        freq="D"
    )

    weekday_dates = observed_dates[
        observed_dates.dayofweek < 5
    ]

    n_weekdays = len(weekday_dates)

    if n_weekdays == 0:
        raise ValueError(
            "No weekdays found in the observation period."
        )

    # ---------------------------------------------------------
    # 4. CREATE ARRIVAL AND DEPARTURE RECORDS
    # ---------------------------------------------------------

    departures = pd.DataFrame({
        "station": df[start_station_column],
        "timestamp": df["journey_start"],
        "movement": -1
    })

    arrivals = pd.DataFrame({
        "station": df[end_station_column],
        "timestamp": df["journey_end"],
        "movement": 1
    })

    movements = pd.concat(
        [departures, arrivals],
        ignore_index=True
    )

    movements["date"] = (
        movements["timestamp"].dt.normalize()
    )

    movements["hour"] = (
        movements["timestamp"].dt.hour
    )

    movements["weekday"] = (
        movements["timestamp"].dt.dayofweek
    )

    movements = movements[
        (movements["weekday"] < 5)
        & (movements["date"].isin(weekday_dates))
    ].copy()

    # ---------------------------------------------------------
    # 5. CALCULATE STATION ACTIVITY
    # ---------------------------------------------------------

    total_activity = (
        movements.groupby("station")
        .size()
        .rename("total_activity")
    )

    average_daily_activity = (
        total_activity / n_weekdays
    ).rename("average_daily_activity")

    # ---------------------------------------------------------
    # 6. CALCULATE MORNING AND EVENING NET FLOWS
    # ---------------------------------------------------------

    morning = movements[
        (movements["hour"] >= morning_hours[0])
        & (movements["hour"] < morning_hours[1])
    ]

    evening = movements[
        (movements["hour"] >= evening_hours[0])
        & (movements["hour"] < evening_hours[1])
    ]

    morning_net = (
        morning.groupby("station")["movement"]
        .sum()
        .div(n_weekdays)
        .rename("morning_net_flow")
    )

    evening_net = (
        evening.groupby("station")["movement"]
        .sum()
        .div(n_weekdays)
        .rename("evening_net_flow")
    )

    # ---------------------------------------------------------
    # 7. COMBINE STATISTICS
    # ---------------------------------------------------------

    station_stats = pd.concat(
        [
            total_activity,
            average_daily_activity,
            morning_net,
            evening_net
        ],
        axis=1
    ).fillna(0)

    station_stats = station_stats[
        station_stats["total_activity"]
        >= min_station_journeys
    ].copy()

    if station_stats.empty:
        raise ValueError(
            "No stations meet the minimum journey threshold."
        )

    # ---------------------------------------------------------
    # 8. CLASSIFY STATIONS
    # ---------------------------------------------------------

    def classify_station(row):

        morning = row["morning_net_flow"]
        evening = row["evening_net_flow"]

        if (
            abs(morning) <= balance_threshold
            and abs(evening) <= balance_threshold
        ):
            return "Relatively balanced"

        if (
            morning < -balance_threshold
            and evening > balance_threshold
        ):
            return "Morning departures / evening arrivals"

        if (
            morning > balance_threshold
            and evening < -balance_threshold
        ):
            return "Morning arrivals / evening departures"

        if (
            morning > balance_threshold
            and evening > balance_threshold
        ):
            return "Net arrivals in both periods"

        if (
            morning < -balance_threshold
            and evening < -balance_threshold
        ):
            return "Net departures in both periods"

        return "Mixed / weak directional pattern"

    station_stats["classification"] = (
        station_stats.apply(
            classify_station,
            axis=1
        )
    )

    # ---------------------------------------------------------
    # 9. CALCULATE CORRELATION
    # ---------------------------------------------------------

    x = station_stats["morning_net_flow"]
    y = station_stats["evening_net_flow"]

    if (
        len(station_stats) >= 2
        and x.std() > 0
        and y.std() > 0
    ):
        correlation = x.corr(y)
    else:
        correlation = np.nan

    # ---------------------------------------------------------
    # 10. SET BUBBLE SIZES
    # ---------------------------------------------------------

    # Square-root scaling reduces the visual dominance
    # of particularly busy stations.

    activity = station_stats["average_daily_activity"]

    if activity.max() > activity.min():

        normalised = (
            (activity - activity.min())
            / (activity.max() - activity.min())
        )

        station_stats["bubble_size"] = (
            min_bubble_size
            + np.sqrt(normalised)
            * (max_bubble_size - min_bubble_size)
        )

    else:
        station_stats["bubble_size"] = (
            (min_bubble_size + max_bubble_size) / 2
        )

    # ---------------------------------------------------------
    # 11. CREATE SCATTER PLOT
    # ---------------------------------------------------------

    colours = {
        "Morning departures / evening arrivals": "#2878B5",
        "Morning arrivals / evening departures": "#D95F02",
        "Net arrivals in both periods": "#2A9D8F",
        "Net departures in both periods": "#8E44AD",
        "Relatively balanced": "#7F8C8D",
        "Mixed / weak directional pattern": "#B0B0B0"
    }

    fig, ax = plt.subplots(figsize=figsize)

    for category, colour in colours.items():

        subset = station_stats[
            station_stats["classification"] == category
        ]

        if subset.empty:
            continue

        ax.scatter(
            subset["morning_net_flow"],
            subset["evening_net_flow"],
            s=subset["bubble_size"],
            color=colour,
            alpha=0.55,
            edgecolors="white",
            linewidths=0.5,
            label=category,
            zorder=3
        )

    # ---------------------------------------------------------
    # 12. APPLY SYMMETRIC LOGARITHMIC SCALES
    # ---------------------------------------------------------

    ax.set_xscale(
        "symlog",
        linthresh=log_linthresh,
        linscale=1,
        base=10
    )

    ax.set_yscale(
        "symlog",
        linthresh=log_linthresh,
        linscale=1,
        base=10
    )

    # ---------------------------------------------------------
    # 13. ZERO AXES
    # ---------------------------------------------------------

    ax.axhline(
        0,
        color="#333333",
        linewidth=1.2,
        zorder=2
    )

    ax.axvline(
        0,
        color="#333333",
        linewidth=1.2,
        zorder=2
    )

    # Symmetrical limits preserve the four-quadrant layout.

    max_abs = max(
        abs(station_stats["morning_net_flow"]).max(),
        abs(station_stats["evening_net_flow"]).max()
    )

    limit = max(
        max_abs * 1.2,
        log_linthresh * 2
    )

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    ax.set_aspect("equal", adjustable="box")

    # ---------------------------------------------------------
    # 14. LABEL DISTINCTIVE STATIONS
    # ---------------------------------------------------------
    
    station_stats["distance_from_origin"] = np.sqrt(
        station_stats["morning_net_flow"] ** 2
        + station_stats["evening_net_flow"] ** 2
    )

    labelled_stations = station_stats.nlargest(
        label_top_n,
        "distance_from_origin"
    )
    '''
    for station_name, row in labelled_stations.iterrows():

        ax.annotate(
            station_name,
            (
                row["morning_net_flow"],
                row["evening_net_flow"]
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="#222222",
            zorder=5
        )
    '''
    # ---------------------------------------------------------
    # 15. TITLES AND FORMATTING
    # ---------------------------------------------------------
    '''
    ax.set_title(
        "Morning and Evening Net Bicycle Flow by Docking Station",
        fontsize=15,
        fontweight="bold",
        pad=18
    )
    '''
    ax.set_xlabel(
        f"Average weekday morning net flow "
        f"({morning_hours[0]:02d}:00–{morning_hours[1]:02d}:00)\n"
        "",
        fontsize=11
    )

    ax.set_ylabel(
        f"Average weekday evening net flow "
        f"({evening_hours[0]:02d}:00–{evening_hours[1]:02d}:00)\n"
        "",
        fontsize=11
    )

    ax.grid(
        True,
        which="major",
        linestyle=":",
        alpha=0.3,
        zorder=0
    )

    ax.grid(
        True,
        which="minor",
        linestyle=":",
        alpha=0.12,
        zorder=0
    )

    ax.set_axisbelow(True)

    # Correlation annotation.
    '''
    if np.isfinite(correlation):
        correlation_text = f"Pearson r = {correlation:.3f}"
    else:
        correlation_text = "Pearson r = N/A"
    
    annotation = (
        f"{correlation_text}\n"
        f"Stations: {len(station_stats)}\n"
        f"Weekdays: {n_weekdays}"
    )
    
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(
            facecolor="white",
            edgecolor="#CCCCCC",
            alpha=0.9,
            boxstyle="round,pad=0.5"
        )
    )
    '''
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
        fontsize=9
    )

    fig.tight_layout()
    
    # ---------------------------------------------------------
    # 16. SAVE FIGURE
    # ---------------------------------------------------------

    output_path = Path(output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight"
    )

    pdf_path = output_path.with_suffix(".pdf")

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    # ---------------------------------------------------------
    # 17. PRINT SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 65)
    print("DOCKING STATION NET FLOW ANALYSIS")
    print("=" * 65)

    print(f"Weekdays analysed: {n_weekdays}")
    print(f"Stations included: {len(station_stats)}")

    print(
        f"Morning period: "
        f"{morning_hours[0]:02d}:00–{morning_hours[1]:02d}:00"
    )

    print(
        f"Evening period: "
        f"{evening_hours[0]:02d}:00–{evening_hours[1]:02d}:00"
    )

    print(f"\nPearson correlation: {correlation:.3f}")

    print("\nStation classifications:")

    print(
        station_stats["classification"]
        .value_counts()
        .to_string()
    )

    print("\nMost distinctive stations:")

    print(
        station_stats[
            [
                "morning_net_flow",
                "evening_net_flow",
                "average_daily_activity",
                "classification"
            ]
        ]
        .loc[labelled_stations.index]
        .round(2)
        .to_string()
    )

    print(f"\nPNG saved to: {output_path}")
    print(f"PDF saved to: {pdf_path}")

    print("=" * 65)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return station_stats

station_stats = plot_station_net_flow_scatter(
    input_file="TestData.csv",
    output_file="station_net_flow_scatter.png",
    duration_column="Total duration (ms)",
    duration_unit="ms",
    morning_hours=(7, 10),
    evening_hours=(16, 19),
    min_station_journeys=20,
    balance_threshold=0.5,
    log_linthresh=0.5,
    label_top_n=8
)

#station_stats.to_csv("station_stats.csv", index=True)