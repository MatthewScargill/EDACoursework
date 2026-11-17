from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter
from shapely.geometry import box


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


def plot_london_usage_heatmap(
    journeys_file="journeys_with_coordinates.csv",
    map_folder="opmplc_gpkg_gb",
    output_name="london_station_usage_heatmap",
    padding=600,
    bandwidth=750,
    resolution=600,
    show_colorbar=False
):
    """
    Create an A4 landscape continuous heatmap of Santander Cycles
    station usage (journey starts + ends).

    Uses the same OS OpenMap - Local basemap styling as
    plot_london_map().

    Parameters
    ----------
    journeys_file : str
        CSV containing journey start/end stations and coordinates.

    map_folder : str
        Folder containing OS OpenMap - Local GeoPackage files.

    output_name : str
        Output filename without extension.

    padding : float
        Padding around all stations in metres.

    bandwidth : float
        Gaussian smoothing standard deviation in metres.

    resolution : int
        Number of grid cells across the map width.

    show_colorbar : bool
        Whether to display a heatmap colour bar.

    Returns
    -------
    fig, ax, station_usage
    """

    # ==================================================
    # 1. LOAD JOURNEY DATA
    # ==================================================

    df = pd.read_csv(journeys_file)

    required_columns = [
        "Start station",
        "End station",
        "latitude_start",
        "longitude_start",
        "latitude_end",
        "longitude_end"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # ==================================================
    # 2. COMBINE START AND END STATIONS
    # ==================================================

    starts = df[
        ["Start station", "latitude_start", "longitude_start"]
    ].copy()

    starts.columns = [
        "station", "latitude", "longitude"
    ]

    ends = df[
        ["End station", "latitude_end", "longitude_end"]
    ].copy()

    ends.columns = [
        "station", "latitude", "longitude"
    ]

    # Every start and end counts as one station use
    all_usage = pd.concat(
        [starts, ends],
        ignore_index=True
    )

    all_usage = all_usage.dropna(
        subset=["station", "latitude", "longitude"]
    )

    # ==================================================
    # 3. COUNT TOTAL USAGE PER STATION
    # ==================================================

    usage_counts = (
        all_usage.groupby("station")
        .size()
        .rename("usage")
        .reset_index()
    )

    # One geographical location per station
    locations = (
        all_usage.drop_duplicates(subset=["station"])
        [["station", "latitude", "longitude"]]
    )

    station_usage = usage_counts.merge(
        locations,
        on="station",
        how="left"
    )

    print(
        f"Stations included: {len(station_usage)}"
    )

    print(
        f"Total recorded station uses: "
        f"{station_usage['usage'].sum():,}"
    )

    # ==================================================
    # 4. CREATE GEODATAFRAME
    # ==================================================

    geo = gpd.GeoDataFrame(
        station_usage,
        geometry=gpd.points_from_xy(
            station_usage["longitude"],
            station_usage["latitude"]
        ),
        crs="EPSG:4326"
    )

    geo = geo.to_crs(epsg=27700)

    # ==================================================
    # 5. CALCULATE EXTENT FROM ALL STATIONS
    # ==================================================

    xmin, ymin, xmax, ymax = geo.total_bounds

    xmin -= padding
    xmax += padding
    ymin -= padding
    ymax += padding

    # ==================================================
    # 6. ADJUST EXTENT FOR A4 LANDSCAPE
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

    print("\nMap extent (EPSG:27700):")
    print(f"X: {xmin:.0f} to {xmax:.0f}")
    print(f"Y: {ymin:.0f} to {ymax:.0f}")

    # ==================================================
    # 7. FIND GEOPACKAGE FILES
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
    # 8. DISCOVER RELEVANT LAYERS
    # ==================================================

    layer_files = []

    # Same keywords as your node map
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
    # 9. CREATE A4 LANDSCAPE FIGURE
    # ==================================================

    fig, ax = plt.subplots(
        figsize=(11.69, 8.27)
    )

    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    # Same edge-to-edge layout as your node map
    fig.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0
    )

    # ==================================================
    # 10. DEFINE MAP STYLING
    # ==================================================

    # Identical colours to your node map
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
    # 11. DEFINE LAYER ORDER
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

    # ==================================================
    # 12. PLOT OS BASEMAP
    # ==================================================

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

            # ------------------------------------------
            # REMOVE UNWANTED BASEMAP POINT FEATURES
            # ------------------------------------------

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
    # 13. CREATE WEIGHTED HEATMAP GRID
    # ==================================================

    # Station coordinates in British National Grid
    x = geo.geometry.x.to_numpy()
    y = geo.geometry.y.to_numpy()

    # Combined starts + ends per station
    weights = geo["usage"].to_numpy()

    nx = resolution

    ny = max(
        2,
        int(resolution * (ymax - ymin) / (xmax - xmin))
    )

    # Place station usage into grid cells
    heat, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=[nx, ny],
        range=[
            [xmin, xmax],
            [ymin, ymax]
        ],
        weights=weights
    )

    # Convert smoothing distance (metres) to grid cells
    cell_size_x = (xmax - xmin) / nx
    cell_size_y = (ymax - ymin) / ny

    sigma_x = bandwidth / cell_size_x
    sigma_y = bandwidth / cell_size_y

    # Smooth into a continuous surface
    heat = gaussian_filter(
        heat,
        sigma=(sigma_x, sigma_y)
    )

    # Transpose for Matplotlib
    heat = heat.T

    # ==================================================
    # 14. PLOT CONTINUOUS HEATMAP
    # ==================================================

    max_density = heat.max()

    # Make very low-density areas transparent
    heat_masked = np.ma.masked_where(
        heat < max_density * 0.02,
        heat
    )

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(alpha=0)

    im = ax.imshow(
        heat_masked,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        cmap=cmap,
        interpolation="bilinear",
        alpha=0.75,
        zorder=10,
        aspect="equal"
    )

    # ==================================================
    # 15. OPTIONAL COLOUR BAR
    # ==================================================

    if show_colorbar:

        cbar = fig.colorbar(
            im,
            ax=ax,
            fraction=0.035,
            pad=0.02
        )

        cbar.set_label(
            "Smoothed station usage density",
            fontsize=10
        )

        cbar.ax.tick_params(labelsize=9)

    # ==================================================
    # 16. FORMAT MAP
    # ==================================================

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_aspect("equal")
    ax.set_axis_off()

    add_north_arrow(ax)

    # No title, matching your updated node map
    # No source text, matching your updated node map

    # ==================================================
    # 17. EXPORT
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

    return fig, ax, station_usage

fig, ax, station_usage = plot_london_usage_heatmap(
    journeys_file="TestData.csv",
    map_folder="opmplc_gpkg_gb",
    output_name="london_station_usage_heatmap",
    padding=600,
    bandwidth=450,
    resolution=600,
    show_colorbar=False
)