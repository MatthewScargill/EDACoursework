import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from pathlib import Path
from matplotlib.collections import LineCollection, PolyCollection


def plot_all_journey_arrows(
    journeys_file="journeys_with_coordinates.csv",
    map_folder="opmplc_gpkg_gb",
    duration_column="Total duration (ms)",
    year=2025,
    output_name="all_santander_journey_arrows",
    padding=600,
    max_duration_minutes=60,
    line_alpha=0.08,
    line_width=0.5,
    arrow_size=65,
    show_colorbar=True
):
    """
    Plot all Santander Cycles journeys as directional arrows.

    Arrow colour represents journey duration.

    The map extent is based on all docking stations,
    matching the previous full-network maps.

    Parameters
    ----------
    duration_column : str
        Journey duration column, assumed to be milliseconds.

    max_duration_minutes : float
        Upper limit of the colour scale.
        Longer journeys are displayed using the maximum colour.

    line_alpha : float
        Transparency of journey lines.

    line_width : float
        Width of journey lines.

    arrow_size : float
        Arrowhead length in metres (EPSG:27700).
    """

    # ==================================================
    # 1. LOAD DATA
    # ==================================================

    df = pd.read_csv(journeys_file)

    required_columns = [
        "Start date",
        "Start station",
        "End station",
        "latitude_start",
        "longitude_start",
        "latitude_end",
        "longitude_end",
        duration_column
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # ==================================================
    # 2. FILTER YEAR AND CLEAN JOURNEYS
    # ==================================================

    df["Start date"] = pd.to_datetime(
        df["Start date"],
        dayfirst=True,
        errors="coerce"
    )

    df = df[
        df["Start date"].dt.year == year
    ].copy()

    df = df.dropna(
        subset=[
            "latitude_start",
            "longitude_start",
            "latitude_end",
            "longitude_end",
            duration_column
        ]
    )

    df[duration_column] = pd.to_numeric(
        df[duration_column],
        errors="coerce"
    )

    df = df[
        df[duration_column] > 0
    ].copy()

    if df.empty:
        raise ValueError("No valid journeys found.")

    # Convert milliseconds to minutes
    df["duration_minutes"] = (
        df[duration_column] / 60000
    )

    # ==================================================
    # 3. CREATE ALL-STATION DATASET
    # ==================================================

    start_stations = df[
        [
            "Start station",
            "latitude_start",
            "longitude_start"
        ]
    ].rename(
        columns={
            "Start station": "station",
            "latitude_start": "latitude",
            "longitude_start": "longitude"
        }
    )

    end_stations = df[
        [
            "End station",
            "latitude_end",
            "longitude_end"
        ]
    ].rename(
        columns={
            "End station": "station",
            "latitude_end": "latitude",
            "longitude_end": "longitude"
        }
    )

    stations = pd.concat(
        [start_stations, end_stations],
        ignore_index=True
    )

    stations = stations.drop_duplicates(
        subset="station"
    )

    stations_gdf = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(
            stations["longitude"],
            stations["latitude"]
        ),
        crs="EPSG:4326"
    ).to_crs("EPSG:27700")

    # ==================================================
    # 4. CALCULATE FULL NETWORK EXTENT
    # ==================================================

    xmin, ymin, xmax, ymax = (
        stations_gdf.total_bounds
    )

    xmin -= padding
    xmax += padding
    ymin -= padding
    ymax += padding

    target_ratio = 11.69 / 8.27

    width = xmax - xmin
    height = ymax - ymin

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    if width / height < target_ratio:
        width = height * target_ratio
    else:
        height = width / target_ratio

    xmin = cx - width / 2
    xmax = cx + width / 2
    ymin = cy - height / 2
    ymax = cy + height / 2

    # ==================================================
    # 5. CONVERT JOURNEY COORDINATES TO EPSG:27700
    # ==================================================

    starts = gpd.GeoSeries(
        gpd.points_from_xy(
            df["longitude_start"],
            df["latitude_start"]
        ),
        crs="EPSG:4326"
    ).to_crs("EPSG:27700")

    ends = gpd.GeoSeries(
        gpd.points_from_xy(
            df["longitude_end"],
            df["latitude_end"]
        ),
        crs="EPSG:4326"
    ).to_crs("EPSG:27700")

    x1 = starts.x.to_numpy()
    y1 = starts.y.to_numpy()

    x2 = ends.x.to_numpy()
    y2 = ends.y.to_numpy()

    # ==================================================
    # 6. CREATE LINE SEGMENTS
    # ==================================================

    segments = np.stack(
        [
            np.column_stack([x1, y1]),
            np.column_stack([x2, y2])
        ],
        axis=1
    )

    # ==================================================
    # 7. SET UP DURATION COLOURS
    # ==================================================

    cmap = plt.get_cmap("YlOrRd")

    norm = mcolors.Normalize(
        vmin=0,
        vmax=max_duration_minutes
    )

    colours = cmap(
        norm(df["duration_minutes"].to_numpy())
    )

    # ==================================================
    # 8. CREATE FIGURE
    # ==================================================

    fig, ax = plt.subplots(
        figsize=(11.69, 8.27)
    )

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    fig.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0
    )

    # ==================================================
    # 9. LOAD OS OPENMAP BASEMAP
    # ==================================================

    gpkg_files = list(
        Path(map_folder).rglob("*.gpkg")
    )

    if not gpkg_files:
        raise FileNotFoundError(
            f"No GeoPackage files found in '{map_folder}'."
        )

    layer_styles = {
        "land": ("#E7EEE5", 1),
        "wood": ("#E7EEE5", 1),
        "surface": ("#E7EEE5", 1),
        "building": ("#EEEEEE", 2),
        "road": ("#D4D4D4", 3),
        "rail": ("#C4C4C4", 4),
        "water": ("#B8D7E8", 5)
    }

    bbox = (xmin, ymin, xmax, ymax)

    for gpkg_file in gpkg_files:

        try:
            available_layers = gpd.list_layers(
                gpkg_file
            )
        except Exception:
            continue

        for layer_name in available_layers["name"]:

            name_lower = layer_name.lower()

            matched_style = None

            for keyword, style in layer_styles.items():

                if keyword in name_lower:
                    matched_style = style
                    break

            if matched_style is None:
                continue

            colour, zorder = matched_style

            try:
                layer = gpd.read_file(
                    gpkg_file,
                    layer=layer_name,
                    bbox=bbox
                )

                if layer.empty or layer.crs is None:
                    continue

                layer = layer.to_crs("EPSG:27700")

                layer = layer[
                    ~layer.geometry.geom_type.isin(
                        ["Point", "MultiPoint"]
                    )
                ]

                if layer.empty:
                    continue

                layer.plot(
                    ax=ax,
                    color=colour,
                    linewidth=0.3,
                    zorder=zorder
                )

            except Exception as e:
                print(f"Skipping {layer_name}: {e}")

    # ==================================================
    # 10. PLOT JOURNEY LINES
    # ==================================================

    line_collection = LineCollection(
        segments,
        colors=colours,
        linewidths=line_width,
        alpha=line_alpha,
        zorder=10
    )

    ax.add_collection(line_collection)

    # ==================================================
    # 11. CREATE DIRECTIONAL ARROWHEADS
    # ==================================================

    dx = x2 - x1
    dy = y2 - y1

    lengths = np.sqrt(dx**2 + dy**2)

    valid = lengths > 0

    ux = np.zeros_like(dx)
    uy = np.zeros_like(dy)

    ux[valid] = dx[valid] / lengths[valid]
    uy[valid] = dy[valid] / lengths[valid]

    # Perpendicular vectors
    px = -uy
    py = ux

    # Arrowhead positioned near destination
    tip_x = x2 - ux * arrow_size * 0.2
    tip_y = y2 - uy * arrow_size * 0.2

    base_x = tip_x - ux * arrow_size
    base_y = tip_y - uy * arrow_size

    half_width = arrow_size * 0.35

    triangles = np.stack(
        [
            np.column_stack([tip_x, tip_y]),
            np.column_stack([
                base_x + px * half_width,
                base_y + py * half_width
            ]),
            np.column_stack([
                base_x - px * half_width,
                base_y - py * half_width
            ])
        ],
        axis=1
    )

    triangles = triangles[valid]

    arrow_collection = PolyCollection(
        triangles,
        facecolors=colours[valid],
        edgecolors="none",
        alpha=line_alpha,
        zorder=11
    )

    ax.add_collection(arrow_collection)

    # ==================================================
    # 12. SET MAP EXTENT
    # ==================================================

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_aspect("equal")
    ax.set_axis_off()

    # ==================================================
    # 13. COLOURBAR
    # ==================================================

    if show_colorbar:

        sm = cm.ScalarMappable(
            norm=norm,
            cmap=cmap
        )

        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=ax,
            fraction=0.025,
            pad=0.015,
            shrink=0.6
        )

        cbar.set_label(
            "Journey duration (minutes)",
            fontsize=10
        )

    # ==================================================
    # 14. EXPORT
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

    print(f"\nJourneys plotted: {len(df):,}")
    print(
        f"Median duration: "
        f"{df['duration_minutes'].median():.1f} minutes"
    )

    return fig, ax, df

fig, ax, journeys = plot_all_journey_arrows(

    journeys_file="TestData.csv",

    map_folder="opmplc_gpkg_gb",

    duration_column="Total duration (ms)",

    year=2025,

    padding=600,

    max_duration_minutes=60,

    line_alpha=0.08,

    line_width=0.5,

    arrow_size=65,

    show_colorbar=False

)