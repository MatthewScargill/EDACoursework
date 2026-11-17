import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("TestData.csv")


def journey_speed_chart(df, max_speed=40, bin_width=1):

    # Create a copy to avoid modifying the original dataframe
    df = df.copy()

    # Convert implied speeds to numeric values
    df["average_speed_kmh"] = pd.to_numeric(
        df["average_speed_kmh"],
        errors="coerce"
    )

    # Remove missing, infinite and negative speeds
    df = df[
        np.isfinite(df["average_speed_kmh"])
        & (df["average_speed_kmh"] >= 0)
    ].copy()

    # Calculate summary statistics using all valid journeys
    mean_speed = df["average_speed_kmh"].mean()
    median_speed = df["average_speed_kmh"].median()

    print(f"Mean implied speed: {mean_speed:.2f} km/h")
    print(f"Median implied speed: {median_speed:.2f} km/h")

    # Select speeds within the displayed range
    # Journeys above max_speed are excluded from the chart only
    plot_df = df[df["average_speed_kmh"] <= max_speed]

    # Create speed bins
    bins = np.arange(0, max_speed + bin_width, bin_width)

    # Plot distribution
    plt.figure(figsize=(12, 6))

    plt.hist(
        plot_df["average_speed_kmh"],
        bins=bins,
        edgecolor="white",
        linewidth=0.3
    )

    plt.xlabel("Implied Average Speed (km/h)")
    plt.ylabel("Number of Journeys")
    plt.title("Distribution of Implied Journey Speeds")

    plt.xticks(np.arange(0, max_speed + 1, 5))

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


journey_speed_chart(df)