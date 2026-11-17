from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import box
from matplotlib.lines import Line2D


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

def plot_london_map(
    stations,
    map_folder="opmplc_gpkg_gb",
    output_name="london_santander_stations",
    padding=1200
):
    """
    Create a professional A4 landscape map of Santander Cycles
    docking stations using OS OpenMap - Local GeoPackage data.

    Parameters
    ----------
    stations : geopandas.GeoDataFrame
        Unique docking station locations with a valid CRS.

    map_folder : str
        Folder containing OS OpenMap - Local GeoPackage files.

    output_name : str
        Output filename without extension.

    padding : float
        Padding around docking stations in metres.

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

    # Convert to British National Grid
    stations = stations.to_crs(epsg=27700).copy()

    # Remove missing or empty geometries
    stations = stations[
        stations.geometry.notna()
        & ~stations.geometry.is_empty
    ].copy()

    if stations.empty:
        raise ValueError(
            "No valid station geometries were found."
        )

    # ==================================================
    # 2. CALCULATE EXTENT FROM ALL STATIONS
    # ==================================================

    xmin, ymin, xmax, ymax = stations.total_bounds

    xmin -= padding
    xmax += padding
    ymin -= padding
    ymax += padding

    # ==================================================
    # 3. ADJUST EXTENT FOR A4 LANDSCAPE
    # ==================================================

    # A4 landscape page: 297 mm x 210 mm
    # Allowing for margins, title and legend
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

    # Bounding box for selecting map features
    london_bbox = box(xmin, ymin, xmax, ymax)

    print("Map extent (EPSG:27700):")
    print(f"X: {xmin:.0f} to {xmax:.0f}")
    print(f"Y: {ymin:.0f} to {ymax:.0f}")

    # ==================================================
    # 4. FIND GEOPACKAGE FILES
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
    # 5. DISCOVER RELEVANT LAYERS
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
    # 6. CREATE A4 LANDSCAPE FIGURE
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
    # 7. DEFINE MAP STYLING
    # ==================================================

    colours = {
        "water": "#B8D7E8",
        "roads": "#D4D4D4",
        "major_roads": "#B5B5B5",
        "green": "#E7EEE5",
        "buildings": "#EEEEEE",
        "rail": "#C4C4C4",
        "stations": "#C0392B"
    }

    # ==================================================
    # 8. LOAD MAP FEATURES
    # ==================================================

    # Plot lower layers first
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

            # Read one feature to identify CRS
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

            # Transform map extent into layer CRS
            bbox_layer = (
                gpd.GeoSeries(
                    [london_bbox],
                    crs="EPSG:27700"
                )
                .to_crs(sample.crs)
                .total_bounds
            )

            # Read only features near the stations
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
                ])].copy()

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
    # 9. PLOT DOCKING STATIONS
    # ==================================================

    stations.plot(
        ax=ax,
        color=colours["stations"],
        markersize=12,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.25,
        zorder=10
    )

    # ==================================================
    # 10. FORMAT MAP
    # ==================================================

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_aspect("equal")
    ax.set_axis_off()

    add_north_arrow(ax)
    '''
    ax.set_title(
        "London Santander Cycles Docking Stations",
        fontsize=16,
        fontweight="bold",
        color="#222222",
        pad=18
    )
    '''
    # ==================================================
    # 11. ADD LEGEND
    # ==================================================

    station_legend = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=colours["stations"],
        markeredgecolor="white",
        markersize=8,
        label="Docking station"
    )

    ax.legend(
        handles=[station_legend],
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#DDDDDD",
        fontsize=10
    )

    # ==================================================
    # 12. ADD DATA SOURCE
    # ==================================================
    '''
    fig.text(
        0.04,
        0.035,
        "Basemap: Ordnance Survey OpenMap Local | "
        "Docking stations: TfL",
        fontsize=8,
        color="#777777"
    )
    '''
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

    plt.show()

    return fig, ax

import pandas as pd
import geopandas as gpd

df = pd.read_csv("journeys_with_coordinates.csv")

starts = df[
    ["Start station", "latitude_start", "longitude_start"]
].copy()

starts.columns = ["station", "latitude", "longitude"]

ends = df[
    ["End station", "latitude_end", "longitude_end"]
].copy()

ends.columns = ["station", "latitude", "longitude"]

stations_df = pd.concat(
    [starts, ends],
    ignore_index=True
)

stations_df = stations_df.dropna(
    subset=["latitude", "longitude"]
)

stations_df = stations_df.drop_duplicates(
    subset=["station"]
)

geo = gpd.GeoDataFrame(
    stations_df,
    geometry=gpd.points_from_xy(
        stations_df["longitude"],
        stations_df["latitude"]
    ),
    crs="EPSG:4326"
)

plot_london_map(
    stations=geo,
    map_folder="opmplc_gpkg_gb",
    output_name="london_santander_stations",
    padding=600
)