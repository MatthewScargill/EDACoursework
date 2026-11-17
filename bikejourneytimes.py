import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("TestData.csv")


def journey_duration_chart(df, max_minutes=120):

    # Work with a copy
    df = df.copy()

    # Convert milliseconds to minutes
    df["Duration (minutes)"] = pd.to_numeric(
        df["Total duration (ms)"],
        errors="coerce"
    ) / 60000

    # Remove missing or invalid durations
    df = df.dropna(subset=["Duration (minutes)"])
    df = df[df["Duration (minutes)"] > 0].copy()

    # Round down to whole minutes
    df["Duration (minutes)"] = np.floor(
        df["Duration (minutes)"]
    ).astype(int)

    # Clamp durations to max_minutes
    df["Duration (minutes)"] = df["Duration (minutes)"].clip(
        upper=max_minutes
    )

    # Count journeys per minute
    counts = df["Duration (minutes)"].value_counts().sort_index()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        counts.index,
        counts.values,
        color="#C0392B",
        width=0.9,
        zorder=3
    )

    ax.set_xlabel("Journey Duration (minutes)")
    ax.set_ylabel("Number of Journeys")
    #ax.set_title("Bike Journey Durations")

    ax.set_xticks(range(0, max_minutes + 1, 5))
    ax.set_xlim(-0.5, max_minutes + 0.5)

    # Horizontal gridlines only
    ax.grid(
        axis="y",
        linestyle="-",
        alpha=0.3,
        zorder=0
    )

    # Remove surrounding box
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove tick marks while keeping labels
    ax.tick_params(axis="both", length=0)

    plt.tight_layout()
    plt.show()


journey_duration_chart(df)