from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import box, LineString
from matplotlib.lines import Line2D


def plot_most_common_journeys(
    df,
    top_n=20,
    period="all",
    combine_directions=True,
    map_folder="opmplc_gpkg_gb",
    output_name="london_most_common_journeys",
    padding=600,
    date_column="Start date"
):
    """
    Plot the most frequently travelled Santander Cycles routes
    on an OS OpenMap Local basemap.

    Prints total journeys and directional journey counts.

    Parameters
    ----------
    df : pandas.DataFrame
        Journey dataset containing station names, coordinates
        and journey start dates.

    top_n : int
        Number of routes to display.

    period : str
        "all"     = All journeys
        "weekday" = Monday to Friday
        "weekend" = Saturday and Sunday

    combine_directions : bool
        True: A -> B and B -> A are combined for ranking and mapping.
        False: Each direction is ranked separately.

    map_folder : str
        Folder containing OS OpenMap Local GeoPackage files.

    output_name : str
        Output filename without extension.

    padding : float
        Map padding in metres.

    date_column : str
        Column containing journey start dates.

    Returns
    -------
    fig, ax, top_routes
    """

    # ==================================================
    # 1. VALIDATE INPUT DATA
    # ==================================================

    period = period.lower()

    if period not in ["all", "weekday", "weekend"]:
        raise ValueError(
            "period must be 'all', 'weekday', or 'weekend'."
        )

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    required_columns = [
        "Start station",
        "End station",
        "latitude_start",
        "longitude_start",
        "latitude_end",
        "longitude_end",
        date_column
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = df[required_columns].copy()

    # ==================================================
    # 2. FILTER BY WEEKDAY / WEEKEND / ALL
    # ==================================================

    data[date_column] = pd.to_datetime(
        data[date_column],
        dayfirst=True,
        errors="coerce"
    )

    data = data.dropna(
        subset=[date_column]
    ).copy()

    data["day_of_week"] = (
        data[date_column].dt.dayofweek
    )

    if period == "weekday":

        data = data[
            data["day_of_week"] < 5
        ].copy()

    elif period == "weekend":

        data = data[
            data["day_of_week"] >= 5
        ].copy()

    print("\n" + "=" * 85)
    print(f"PERIOD SELECTED: {period.upper()}")
    print(f"Journeys in selected period: {len(data):,}")
    print("=" * 85)

    # ==================================================
    # 3. CLEAN COORDINATES
    # ==================================================

    coordinate_columns = [
        "latitude_start",
        "longitude_start",
        "latitude_end",
        "longitude_end"
    ]

    for col in coordinate_columns:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna(
        subset=[
            "Start station",
            "End station"
        ] + coordinate_columns
    ).copy()

    if data.empty:
        raise ValueError(
            f"No valid journeys found for period '{period}'."
        )

    # ==================================================
    # 4. COUNT JOURNEYS IN EACH DIRECTION
    # ==================================================

    directional_counts = (
        data.groupby(
            ["Start station", "End station"]
        )
        .size()
        .reset_index(name="directional_count")
    )

    direction_lookup = {
        (
            row["Start station"],
            row["End station"]
        ): int(row["directional_count"])

        for _, row in directional_counts.iterrows()
    }

    # ==================================================
    # 5. COUNT ROUTES
    # ==================================================

    if combine_directions:

        # Sort station pairs so A-B and B-A are identical
        station_pairs = np.sort(
            data[
                ["Start station", "End station"]
            ].to_numpy(dtype=str),
            axis=1
        )

        data["station_a"] = station_pairs[:, 0]
        data["station_b"] = station_pairs[:, 1]

        route_counts = (
            data.groupby(
                ["station_a", "station_b"]
            )
            .size()
            .reset_index(name="journey_count")
        )

    else:

        route_counts = directional_counts.rename(
            columns={
                "Start station": "station_a",
                "End station": "station_b",
                "directional_count": "journey_count"
            }
        )

    # Exclude same-station journeys
    route_counts = route_counts[
        route_counts["station_a"]
        != route_counts["station_b"]
    ].copy()

    top_routes = (
        route_counts.sort_values(
            "journey_count",
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    if top_routes.empty:
        raise ValueError(
            "No valid station-to-station routes were found."
        )

    # ==================================================
    # 6. PRINT MOST COMMON ROUTES AND DIRECTIONS
    # ==================================================

    print(
        f"\nTOP {len(top_routes)} MOST COMMON ROUTES "
        f"({period.upper()})"
    )

    print("=" * 85)

    for i, row in top_routes.iterrows():

        station_a = row["station_a"]
        station_b = row["station_b"]

        total = int(row["journey_count"])

        if combine_directions:

            a_to_b = direction_lookup.get(
                (station_a, station_b),
                0
            )

            b_to_a = direction_lookup.get(
                (station_b, station_a),
                0
            )

            a_to_b_pct = a_to_b / total * 100
            b_to_a_pct = b_to_a / total * 100

            print(
                f"\n{i + 1}. {station_a} ↔ {station_b}"
            )

            print(
                f"   Total journeys: {total:,}"
            )

            print(
                f"   {station_a} → {station_b}: "
                f"{a_to_b:,} ({a_to_b_pct:.1f}%)"
            )

            print(
                f"   {station_b} → {station_a}: "
                f"{b_to_a:,} ({b_to_a_pct:.1f}%)"
            )

        else:

            print(
                f"\n{i + 1}. {station_a} → {station_b}"
            )

            print(
                f"   Journeys: {total:,}"
            )

    # ==================================================
    # 7. BUILD STATION COORDINATE LOOKUP
    # ==================================================

    starts = data[
        [
            "Start station",
            "latitude_start",
            "longitude_start"
        ]
    ].copy()

    starts.columns = [
        "station",
        "latitude",
        "longitude"
    ]

    ends = data[
        [
            "End station",
            "latitude_end",
            "longitude_end"
        ]
    ].copy()

    ends.columns = [
        "station",
        "latitude",
        "longitude"
    ]

    station_locations = pd.concat(
        [starts, ends],
        ignore_index=True
    )

    station_locations = (
        station_locations.groupby("station")[
            ["latitude", "longitude"]
        ]
        .median()
        .reset_index()
    )

    stations = gpd.GeoDataFrame(
        station_locations,
        geometry=gpd.points_from_xy(
            station_locations["longitude"],
            station_locations["latitude"]
        ),
        crs="EPSG:4326"
    ).to_crs(epsg=27700)

    coordinate_lookup = {
        row["station"]: row.geometry
        for _, row in stations.iterrows()
    }

    # ==================================================
    # 8. CREATE ROUTE GEOMETRIES
    # ==================================================

    route_geometries = []
    valid_route_rows = []

    for _, row in top_routes.iterrows():

        start = coordinate_lookup.get(
            row["station_a"]
        )

        end = coordinate_lookup.get(
            row["station_b"]
        )

        if start is None or end is None:
            continue

        if start.equals(end):
            continue

        route_geometries.append(
            LineString([start, end])
        )

        valid_route_rows.append(
            row.to_dict()
        )

    if not route_geometries:
        raise ValueError(
            "No valid route geometries could be created."
        )

    routes = gpd.GeoDataFrame(
        valid_route_rows,
        geometry=route_geometries,
        crs="EPSG:27700"
    )

    # ==================================================
    # 9. CALCULATE MAP EXTENT
    # ==================================================

    xmin, ymin, xmax, ymax = stations.total_bounds

    xmin -= padding
    xmax += padding
    ymin -= padding
    ymax += padding

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

    london_bbox = box(
        xmin, ymin, xmax, ymax
    )

    # ==================================================
    # 10. FIND GEOPACKAGE FILES
    # ==================================================

    gpkg_files = sorted(
        Path(map_folder).rglob("*.gpkg")
    )

    if not gpkg_files:
        raise FileNotFoundError(
            f"No GeoPackage files found in '{map_folder}'."
        )

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

            if any(
                word in layer_name.lower()
                for word in keywords
            ):
                layer_files.append(
                    (file, layer_name)
                )

    # ==================================================
    # 11. CREATE A4 LANDSCAPE FIGURE
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

    colours = {
        "water": "#B8D7E8",
        "roads": "#D4D4D4",
        "green": "#E7EEE5",
        "buildings": "#EEEEEE",
        "rail": "#C4C4C4",
        "routes": "#C0392B",
        "stations": "#922B21"
    }

    # ==================================================
    # 12. LOAD BASEMAP FEATURES
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
                continue

            bbox_layer = (
                gpd.GeoSeries(
                    [london_bbox],
                    crs="EPSG:27700"
                )
                .to_crs(sample.crs)
                .total_bounds
            )

            layer_data = gpd.read_file(
                file,
                layer=layer,
                bbox=tuple(bbox_layer)
            )

            if layer_data.empty:
                continue

            layer_data = layer_data[
                ~layer_data.geometry.geom_type.isin([
                    "Point",
                    "MultiPoint"
                ])
            ].copy()

            if layer_data.empty:
                continue

            layer_data = layer_data.to_crs(
                epsg=27700
            )

            if (
                "land" in layer_lower
                or "wood" in layer_lower
            ):

                layer_data.plot(
                    ax=ax,
                    color=colours["green"],
                    edgecolor="none",
                    zorder=1
                )

            elif "building" in layer_lower:

                layer_data.plot(
                    ax=ax,
                    color=colours["buildings"],
                    edgecolor="none",
                    zorder=2
                )

            elif "road" in layer_lower:

                layer_data.plot(
                    ax=ax,
                    color=colours["roads"],
                    linewidth=0.5,
                    zorder=3
                )

            elif "rail" in layer_lower:

                layer_data.plot(
                    ax=ax,
                    color=colours["rail"],
                    linewidth=0.6,
                    zorder=4
                )

            elif "water" in layer_lower:

                layer_data.plot(
                    ax=ax,
                    color=colours["water"],
                    edgecolor="none",
                    linewidth=1.5,
                    zorder=5
                )

        except Exception as error:
            print(
                f"Could not plot {layer}: {error}"
            )

    # ==================================================
    # 13. PLOT ROUTES
    # ==================================================

    min_count = routes["journey_count"].min()
    max_count = routes["journey_count"].max()

    if max_count == min_count:

        routes["line_width"] = 3.0

    else:

        routes["line_width"] = (
            1.5
            + 5.0
            * (
                routes["journey_count"] - min_count
            )
            / (max_count - min_count)
        )

    routes = routes.sort_values(
        "journey_count",
        ascending=True
    )

    for _, route in routes.iterrows():

        gpd.GeoSeries(
            [route.geometry],
            crs="EPSG:27700"
        ).plot(
            ax=ax,
            color=colours["routes"],
            linewidth=route["line_width"],
            alpha=0.75,
            zorder=10
        )

    # ==================================================
    # 14. PLOT ROUTE ENDPOINTS
    # ==================================================

    selected_station_names = set(
        routes["station_a"]
    ) | set(
        routes["station_b"]
    )

    selected_stations = stations[
        stations["station"].isin(
            selected_station_names
        )
    ]

    selected_stations.plot(
        ax=ax,
        color=colours["stations"],
        markersize=22,
        edgecolor="white",
        linewidth=0.5,
        zorder=12
    )

    # ==================================================
    # 15. FORMAT MAP
    # ==================================================

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_aspect("equal")
    ax.set_axis_off()

    # North arrow
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

    # ==================================================
    # 16. ADD LEGEND
    # ==================================================

    legend_handles = [
        Line2D(
            [0], [0],
            color=colours["routes"],
            linewidth=2,
            label="Less frequent route"
        ),
        Line2D(
            [0], [0],
            color=colours["routes"],
            linewidth=6,
            label="More frequent route"
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=colours["stations"],
            markeredgecolor="white",
            markersize=7,
            label="Route endpoint"
        )
    ]

    ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#DDDDDD",
        fontsize=10
    )

    # ==================================================
    # 17. EXPORT MAP
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

    return fig, ax, top_routes


# ======================================================
# RUN THE FUNCTION
# ======================================================

df = pd.read_csv("TestData.csv")

fig, ax, top_routes = plot_most_common_journeys(
    df=df,
    top_n=400,
    period="weekday",  # "all", "weekday", or "weekend"
    combine_directions=True,
    map_folder="opmplc_gpkg_gb",
    output_name="london_top_20_routes_weekday_2025",
    padding=600
)