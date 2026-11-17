import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_marathon_route_delays(
    df,
    comparison_date="2025-04-06",
    marathon_date="2025-04-27",
    date_column="Start date",
    start_column="Start station",
    end_column="End station",
    duration_column="duration_minutes",
    min_journeys=5,
    top_n=15,
    output_name="marathon_route_delays"
):
    """
    Identify station-to-station connections with the largest
    increase in median journey duration on Marathon Day.

    Connections are directional:
        Station A -> Station B is different from B -> A.

    Only includes connections with at least min_journeys
    on BOTH dates.

    Saves:
        output_name.pdf
        output_name.png

    Returns:
        DataFrame containing comparison statistics for all
        qualifying station-to-station connections.
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
        raise ValueError(
            "No journey duration column found."
        )

    data = data.dropna(
        subset=[
            date_column,
            start_column,
            end_column,
            "_duration_min"
        ]
    )

    data = data[data["_duration_min"] > 0].copy()

    data["_date"] = data[date_column].dt.normalize()

    comparison_date = pd.Timestamp(
        comparison_date
    ).normalize()

    marathon_date = pd.Timestamp(
        marathon_date
    ).normalize()

    # --------------------------------------------------
    # Select relevant dates
    # --------------------------------------------------

    comparison = data[
        data["_date"] == comparison_date
    ].copy()

    marathon = data[
        data["_date"] == marathon_date
    ].copy()

    if comparison.empty or marathon.empty:
        raise ValueError(
            "No journeys found for one or both dates."
        )

    # --------------------------------------------------
    # Group by directional station connection
    # --------------------------------------------------

    group_columns = [
        start_column,
        end_column
    ]

    def summarise_connections(journeys):

        return (
            journeys.groupby(group_columns)["_duration_min"]
            .agg(
                journeys="count",
                median_duration="median",
                mean_duration="mean"
            )
            .reset_index()
        )

    comparison_stats = summarise_connections(
        comparison
    )

    marathon_stats = summarise_connections(
        marathon
    )

    # --------------------------------------------------
    # Match connections across dates
    # --------------------------------------------------

    results = comparison_stats.merge(
        marathon_stats,
        on=group_columns,
        how="inner",
        suffixes=("_comparison", "_marathon")
    )

    # Require sufficient journeys on both dates
    results = results[
        (results["journeys_comparison"] >= min_journeys)
        &
        (results["journeys_marathon"] >= min_journeys)
    ].copy()

    if results.empty:
        raise ValueError(
            "No connections meet the minimum journey "
            "threshold on both dates. Try reducing "
            "min_journeys."
        )

    # --------------------------------------------------
    # Calculate duration changes
    # --------------------------------------------------

    results["median_increase_min"] = (
        results["median_duration_marathon"]
        - results["median_duration_comparison"]
    )

    results["median_change_pct"] = (
        results["median_increase_min"]
        / results["median_duration_comparison"]
        * 100
    )

    results["mean_increase_min"] = (
        results["mean_duration_marathon"]
        - results["mean_duration_comparison"]
    )

    results["connection"] = (
        results[start_column].astype(str)
        + " → "
        + results[end_column].astype(str)
    )

    # Sort by largest increase in median duration
    results = results.sort_values(
        "median_increase_min",
        ascending=False
    ).reset_index(drop=True)

    # Only show connections that became slower
    top_connections = (
        results[
            results["median_increase_min"] > 0
        ]
        .head(top_n)
        .sort_values("median_increase_min")
    )

    if top_connections.empty:
        print(
            "No qualifying connections had an increase "
            "in median duration."
        )
        return results

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    fig_height = max(
        5,
        len(top_connections) * 0.48 + 1.5
    )

    fig, ax = plt.subplots(
        figsize=(11, fig_height)
    )

    red = "#C0392B"

    y = np.arange(len(top_connections))

    bars = ax.barh(
        y,
        top_connections["median_increase_min"],
        color=red,
        height=0.65,
        zorder=3
    )

    # --------------------------------------------------
    # Labels
    # --------------------------------------------------

    ax.set_yticks(y)

    ax.set_yticklabels(
        top_connections["connection"],
        fontsize=9
    )

    # Show median increase and sample sizes
    for bar, (_, row) in zip(
        bars,
        top_connections.iterrows()
    ):

        increase = row["median_increase_min"]

        n_comparison = int(
            row["journeys_comparison"]
        )

        n_marathon = int(
            row["journeys_marathon"]
        )

        ax.text(
            increase + 0.15,
            bar.get_y() + bar.get_height() / 2,
            (
                f"+{increase:.1f} min "
                f"(n={n_comparison}/{n_marathon})"
            ),
            va="center",
            ha="left",
            fontsize=8
        )

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------

    ax.set_xlabel(
        "Increase in median journey duration (minutes)",
        fontsize=11
    )

    ax.set_xlim(
        0,
        top_connections["median_increase_min"].max() * 1.55
    )

    ax.grid(
        axis="x",
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
        length=0
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
    # Print results
    # --------------------------------------------------

    print("\nCONNECTIONS WITH LARGEST DURATION INCREASES")
    print("-" * 65)

    print(
        top_connections[
            [
                "connection",
                "journeys_comparison",
                "journeys_marathon",
                "median_duration_comparison",
                "median_duration_marathon",
                "median_increase_min",
                "median_change_pct"
            ]
        ]
        .sort_values(
            "median_increase_min",
            ascending=False
        )
        .to_string(index=False)
    )

    print(
        f"\nQualifying connections: {len(results):,}"
    )

    return results


# --------------------------------------------------
# Run
# --------------------------------------------------

df = pd.read_csv("TestData.csv")

route_delays = plot_marathon_route_delays(
    df,
    comparison_date="2025-04-06",
    marathon_date="2025-04-27",
    min_journeys=5,
    top_n=15
)
