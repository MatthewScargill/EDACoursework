import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import calendar


def plot_monthly_cycle_usage(
    journeys_file="journeys_with_coordinates.csv",
    date_column="Start date",
    year=2025,
    output_name="monthly_santander_usage",
    show_title=False
):
    """
    Create a professional bar chart showing monthly
    Santander Cycles journey frequency.

    Parameters
    ----------
    journeys_file : str
        CSV containing Santander Cycles journey data.

    date_column : str
        Column containing journey start dates.

    year : int
        Year to analyse.

    output_name : str
        Output filename without extension.

    show_title : bool
        Whether to display a chart title.

    Returns
    -------
    fig, ax, monthly_usage
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
    # 2. PROCESS DATES
    # ==================================================

    df[date_column] = pd.to_datetime(
        df[date_column],
        dayfirst=True,
        errors="coerce"
    )

    # Keep journeys from the selected year
    df = df[
        df[date_column].dt.year == year
    ].copy()

    if df.empty:
        raise ValueError(
            f"No journeys found for {year}."
        )

    # ==================================================
    # 3. COUNT JOURNEYS PER MONTH
    # ==================================================

    monthly_usage = (
        df.groupby(
            df[date_column].dt.month
        )
        .size()
        .reindex(
            range(1, 13),
            fill_value=0
        )
    )

    monthly_usage.index = [
        calendar.month_abbr[i]
        for i in range(1, 13)
    ]

    monthly_usage.name = "journeys"

    print(f"\nSantander Cycles usage in {year}")
    print(monthly_usage)

    print(
        f"\nTotal journeys: "
        f"{monthly_usage.sum():,}"
    )

    # ==================================================
    # 4. CREATE FIGURE
    # ==================================================

    fig, ax = plt.subplots(
        figsize=(11.69, 8.27)
    )

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    fig.subplots_adjust(
        left=0.11,
        right=0.96,
        top=0.93,
        bottom=0.13
    )

    # ==================================================
    # 5. PLOT MONTHLY BARS
    # ==================================================

    bars = ax.bar(
        monthly_usage.index,
        monthly_usage.values,
        color="#C0392B",
        width=0.68,
        edgecolor="none",
        zorder=3
    )

    # ==================================================
    # 6. FORMAT AXES
    # ==================================================

    ax.set_xlabel(
        "Month",
        fontsize=12,
        labelpad=12
    )

    ax.set_ylabel(
        "Number of journeys",
        fontsize=12,
        labelpad=12
    )

    ax.tick_params(
        axis="both",
        labelsize=11,
        length=0,
        pad=8
    )

    # Format large numbers with commas
    ax.yaxis.set_major_formatter(
        ticker.StrMethodFormatter("{x:,.0f}")
    )

    # Start y-axis at zero
    ax.set_ylim(
        0,
        monthly_usage.max() * 1.12
    )

    # ==================================================
    # 7. ADD SUBTLE GRIDLINES
    # ==================================================

    ax.grid(
        axis="y",
        color="#E6E6E6",
        linewidth=0.8,
        zorder=0
    )

    ax.grid(
        axis="x",
        visible=False
    )

    # ==================================================
    # 8. REMOVE UNNECESSARY BORDERS
    # ==================================================

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.spines["bottom"].set_color(
        "#CCCCCC"
    )

    # ==================================================
    # 9. OPTIONAL TITLE
    # ==================================================

    if show_title:

        ax.set_title(
            f"Monthly Santander Cycles Usage, {year}",
            fontsize=16,
            fontweight="bold",
            color="#222222",
            pad=20
        )

    # ==================================================
    # 10. EXPORT
    # ==================================================

    png_file = f"{output_name}.png"
    pdf_file = f"{output_name}.pdf"

    fig.savefig(
        png_file,
        dpi=600,
        facecolor="white"
    )

    fig.savefig(
        pdf_file,
        facecolor="white"
    )

    print("\nChart exported successfully:")
    print(f"  PNG: {png_file}")
    print(f"  PDF: {pdf_file}")

    plt.show()

    return fig, ax, monthly_usage



fig, ax, monthly_usage = plot_monthly_cycle_usage(
    journeys_file="TestData.csv",
    date_column="Start date",
    year=2025,
    output_name="monthly_santander_usage",
    show_title=False
)