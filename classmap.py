from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import box
from matplotlib.lines import Line2D


# ============================================================
# NORTH ARROW
# ============================================================

def add_north_arrow(ax):
    """
    Add a minimalist north arrow in the bottom-right
    corner of a Matplotlib map.
    """

    ax.annotate(
        "N",
        xy=(0.955, 0.11),
        xytext=(0.955, 0.045),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#333333",
        arrowprops=dict(
            arrowstyle="-|>",
            color="#333333",
            lw=1.5,
            mutation_scale=15
        ),
        zorder=20
    )


# ============================================================
# STATION CLASSIFICATION MAP
# ============================================================

def plot_london_classification_map(
    stations,
    map_folder="opmplc_gpkg_gb",
    output_name="london_station_classifications",
    padding=1200,
    marker_size=18
):
    """
    Create an A4 landscape map showing Santander Cycles
    docking stations coloured by morning/evening net-flow
    classification.

    Uses the same OS OpenMap Local basemap and styling
    as the original London docking station map.

    Parameters
    ----------
    stations : geopandas.GeoDataFrame
        Unique docking station locations containing a
        'classification' column and valid CRS.

    map_folder : str
        Folder containing OS OpenMap Local GeoPackages.

    output_name : str
        Output filename without extension.

    padding : float
        Padding around docking stations in metres.

    marker_size : float
        Size of station markers.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """

    # ==================================================
    # 1. CHECK INPUT DATA
    # ==================================================

    if stations is None or stations.empty:
        raise ValueError(
            "The stations GeoDataFrame is empty."
        )

    if stations.crs is None:
        raise ValueError(
            "The stations GeoDataFrame must have a CRS."
        )

    if "classification" not in stations.columns:
        raise ValueError(
            "The stations GeoDataFrame must contain "
            "a 'classification' column."
        )

    stations = stations.to_crs(epsg=27700).copy()

    stations = stations[
        stations.geometry.notna()
        & ~stations.geometry.is_empty
    ].copy()

    stations = stations.dropna(
        subset=["classification"]
    )

    if stations.empty:
        raise ValueError(
            "No valid classified stations were found."
        )

    # ==================================================
    # 2. DEFINE CLASSIFICATION COLOURS
    # ==================================================

    # Identical to the scatter plot colours.

    classification_colours = {
        "Morning departures / evening arrivals": "#2878B5",
        "Morning arrivals / evening departures": "#D95F02",
        "Net arrivals in both periods": "#2A9D8F",
        "Net departures in both periods": "#8E44AD",
        "Relatively balanced": "#7F8C8D",
        "Mixed / weak directional pattern": "#B0B0B0"
    }

    unknown_classes = (
        set(stations["classification"].unique())
        - set(classification_colours)
    )

    if unknown_classes:
        raise ValueError(
            "Unknown station classifications: "
            f"{sorted(unknown_classes)}"
        )

    # ==================================================
    # 3. CALCULATE EXTENT FROM ALL STATIONS
    # ==================================================

    xmin, ymin, xmax, ymax = stations.total_bounds

    xmin -= padding
    xmax += padding
    ymin -= padding
    ymax += padding

    # ==================================================
    # 4. ADJUST EXTENT FOR A4 LANDSCAPE
    # ==================================================

    target_aspect = 1.45

    width = xmax - xmin
    height = ymax - ymin

    centre_x = (xmin + xmax) / 2
    centre_y = (ymin + ymax) / 2

    if width / height < target_aspect:
        width = height * target_aspect
    else:
        height = width / target_aspect

    xmin = centre_x - width / 2
    xmax = centre_x + width / 2

    ymin = centre_y - height / 2
    ymax = centre_y + height / 2

    london_bbox = box(xmin, ymin, xmax, ymax)

    print("Map extent (EPSG:27700):")
    print(f"X: {xmin:.0f} to {xmax:.0f}")
    print(f"Y: {ymin:.0f} to {ymax:.0f}")

    # ==================================================
    # 5. FIND GEOPACKAGE FILES
    # ==================================================

    gpkg_files = sorted(
        Path(map_folder).rglob("*.gpkg")
    )

    if not gpkg_files:
        raise FileNotFoundError(
            f"No GeoPackage files found in '{map_folder}'."
        )

    print(
        f"\nFound {len(gpkg_files)} GeoPackage file(s)."
    )

    # ==================================================
    # 6. DISCOVER RELEVANT LAYERS
    # ==================================================

    layer_files = []

    keywords = [
        "road",
        "water",
        "surface",
        "wood",
        "land",
        "building",
        "rail"
    ]

    for file in gpkg_files:

        try:
            layers = gpd.list_layers(file)

        except Exception as error:
            print(f"Could not inspect {file}: {error}")
            continue

        for _, row in layers.iterrows():

            layer_name = row["name"]
            layer_lower = layer_name.lower()

            if any(
                word in layer_lower
                for word in keywords
            ):
                layer_files.append(
                    (file, layer_name)
                )

    print("\nRelevant layers found:")

    for file, layer in layer_files:
        print(f"  {file.name} → {layer}")

    # ==================================================
    # 7. CREATE A4 LANDSCAPE FIGURE
    # ==================================================

    fig, ax = plt.subplots(
        figsize=(11.69, 8.27)
    )

    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    fig.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0
    )

    # ==================================================
    # 8. DEFINE BASEMAP STYLING
    # ==================================================

    colours = {
        "water": "#B8D7E8",
        "roads": "#D4D4D4",
        "major_roads": "#B5B5B5",
        "green": "#E7EEE5",
        "buildings": "#EEEEEE",
        "rail": "#C4C4C4"
    }

    # ==================================================
    # 9. LOAD MAP FEATURES
    # ==================================================

    def layer_priority(name):

        name = name.lower()

        if "land" in name or "wood" in name:
            return 1

        if "building" in name:
            return 2

        if "road" in name:
            return 3

        if "rail" in name:
            return 4

        if "water" in name:
            return 5

        return 6

    layer_files.sort(
        key=lambda item: layer_priority(item[1])
    )

    for file, layer in layer_files:

        layer_lower = layer.lower()

        try:

            sample = gpd.read_file(
                file,
                layer=layer,
                rows=1
            )

            if sample.crs is None:
                print(
                    f"Skipping {layer}: CRS unavailable."
                )
                continue

            bbox_layer = (
                gpd.GeoSeries(
                    [london_bbox],
                    crs="EPSG:27700"
                )
                .to_crs(sample.crs)
                .total_bounds
            )

            data = gpd.read_file(
                file,
                layer=layer,
                bbox=tuple(bbox_layer)
            )

            if data.empty:
                continue

            data = data[
                ~data.geometry.geom_type.isin([
                    "Point",
                    "MultiPoint"
                ])
            ].copy()

            if data.empty:
                continue

            data = data.to_crs(epsg=27700)

            # ------------------------------------------
            # GREEN SPACES
            # ------------------------------------------

            if (
                "land" in layer_lower
                or "wood" in layer_lower
            ):

                data.plot(
                    ax=ax,
                    color=colours["green"],
                    edgecolor="none",
                    zorder=1
                )

            # ------------------------------------------
            # BUILDINGS
            # ------------------------------------------

            elif "building" in layer_lower:

                data.plot(
                    ax=ax,
                    color=colours["buildings"],
                    edgecolor="none",
                    zorder=2
                )

            # ------------------------------------------
            # ROADS
            # ------------------------------------------

            elif "road" in layer_lower:

                data.plot(
                    ax=ax,
                    color=colours["roads"],
                    linewidth=0.5,
                    zorder=3
                )

            # ------------------------------------------
            # RAILWAYS
            # ------------------------------------------

            elif "rail" in layer_lower:

                data.plot(
                    ax=ax,
                    color=colours["rail"],
                    linewidth=0.6,
                    zorder=4
                )

            # ------------------------------------------
            # WATER
            # ------------------------------------------

            elif "water" in layer_lower:

                data.plot(
                    ax=ax,
                    color=colours["water"],
                    edgecolor="none",
                    linewidth=1.5,
                    zorder=5
                )

            print(
                f"Plotted {layer}: {len(data)} features"
            )

        except Exception as error:

            print(
                f"Could not plot {layer}: {error}"
            )

    # ==================================================
    # 10. PLOT STATIONS BY CLASSIFICATION
    # ==================================================

    for classification, colour in classification_colours.items():

        subset = stations[
            stations["classification"] == classification
        ]

        if subset.empty:
            continue

        subset.plot(
            ax=ax,
            color=colour,
            markersize=marker_size,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.25,
            zorder=10
        )

    # ==================================================
    # 11. FORMAT MAP
    # ==================================================

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_aspect("equal")
    ax.set_axis_off()

    add_north_arrow(ax)

    # ==================================================
    # 12. ADD CLASSIFICATION LEGEND
    # ==================================================

    legend_handles = []

    for classification, colour in classification_colours.items():

        count = (
            stations["classification"] == classification
        ).sum()

        if count == 0:
            continue

        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=colour,
                markeredgecolor="white",
                markersize=8,
                linestyle="None",
                label=f"{classification} ({count})"
            )
        )

    ax.legend(
        handles=legend_handles,
        title="Morning / evening net-flow pattern",
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#DDDDDD",
        fontsize=8,
        title_fontsize=9
    )

    # ==================================================
    # 13. EXPORT MAP
    # ==================================================

    png_file = f"{output_name}.png"
    pdf_file = f"{output_name}.pdf"

    fig.savefig(
        png_file,
        dpi=300,
        facecolor="white"
    )

    fig.savefig(
        pdf_file,
        facecolor="white"
    )

    print("\nMap exported successfully:")
    print(f"  PNG: {png_file}")
    print(f"  PDF: {pdf_file}")

    print("\nClassification counts:")

    print(
        stations["classification"]
        .value_counts()
        .to_string()
    )

    plt.show()

    return fig, ax


# ============================================================
# LOAD STATION COORDINATES
# ============================================================

# Same filename as your original mapping script.

df = pd.read_csv("journeys_with_coordinates.csv")

starts = df[
    ["Start station", "latitude_start", "longitude_start"]
].copy()

starts.columns = [
    "station",
    "latitude",
    "longitude"
]

ends = df[
    ["End station", "latitude_end", "longitude_end"]
].copy()

ends.columns = [
    "station",
    "latitude",
    "longitude"
]

stations_df = pd.concat(
    [starts, ends],
    ignore_index=True
)

stations_df["station"] = (
    stations_df["station"]
    .astype("string")
    .str.strip()
)

stations_df["latitude"] = pd.to_numeric(
    stations_df["latitude"],
    errors="coerce"
)

stations_df["longitude"] = pd.to_numeric(
    stations_df["longitude"],
    errors="coerce"
)

stations_df = stations_df.dropna(
    subset=[
        "station",
        "latitude",
        "longitude"
    ]
)

# Use median coordinates if the same station appears
# repeatedly in the journey dataset.

stations_df = (
    stations_df
    .groupby("station", as_index=False)[
        ["latitude", "longitude"]
    ]
    .median()
)


# ============================================================
# LOAD CLASSIFICATIONS FROM SCATTER PLOT
# ============================================================

# Created using:
# station_stats.to_csv("station_stats.csv", index=True)

station_stats = pd.read_csv(
    "station_stats.csv",
    index_col=0
)

station_stats.index = (
    station_stats.index
    .astype("string")
    .str.strip()
)

station_stats.index.name = "station"

# Only the classification is required for this map.

classifications = (
    station_stats[["classification"]]
    .reset_index()
)


# ============================================================
# MERGE COORDINATES WITH CLASSIFICATIONS
# ============================================================

stations_df = stations_df.merge(
    classifications,
    on="station",
    how="inner",
    validate="one_to_one"
)

print(
    f"\nMatched {len(stations_df)} classified stations "
    "to coordinates."
)

if stations_df.empty:
    raise ValueError(
        "No station classifications matched the "
        "journey coordinate data."
    )


# ============================================================
# CREATE GEODATAFRAME
# ============================================================

geo = gpd.GeoDataFrame(
    stations_df,
    geometry=gpd.points_from_xy(
        stations_df["longitude"],
        stations_df["latitude"]
    ),
    crs="EPSG:4326"
)


# ============================================================
# GENERATE MAP
# ============================================================

plot_london_classification_map(
    stations=geo,
    map_folder="opmplc_gpkg_gb",
    output_name="london_station_classifications",
    padding=600,
    marker_size=18
)