import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pathlib import Path
from shapely.geometry import box
from matplotlib.collections import LineCollection, PolyCollection
from pyproj import Transformer


def plot_comparative_journey_arrows(
    date_left,
    date_right,
    journeys_file="journeys_with_coordinates.csv",
    map_folder="opmplc_gpkg_gb",
    date_column="Start date",
    duration_column="Total duration (ms)",
    duration_unit="ms",
    output_name=None,
    padding=600,
    max_duration_minutes=60,
    line_alpha=0.15,
    line_width=0.7,
    arrow_size=65,
    show_colorbar=True,
    show_stations=False,
    show_titles=True
):
    """
    Plot Santander Cycles journeys on two selected dates.

    Features
    --------
    - Two maps side by side
    - OS OpenMap – Local basemap beneath both maps
    - Directional journey arrows
    - Journey duration represented by colour
    - Identical geographical extent
    - One shared colourbar
    - Automatically calculated figure height
    - Minimal white space
    - Fixed A4 landscape width (11.69 inches)

    Dates must be formatted YYYY-MM-DD.

    Returns
    -------
    fig, axes, comparison_data
    """

    # ==================================================
    # 1. VALIDATE INPUTS
    # ==================================================

    left_date = pd.to_datetime(
        date_left,
        format="%Y-%m-%d"
    ).normalize()

    right_date = pd.to_datetime(
        date_right,
        format="%Y-%m-%d"
    ).normalize()

    conversion = {
        "ms": 60000,
        "seconds": 60,
        "minutes": 1
    }

    if duration_unit not in conversion:
        raise ValueError(
            "duration_unit must be 'ms', 'seconds', or 'minutes'."
        )

    if max_duration_minutes <= 0:
        raise ValueError(
            "max_duration_minutes must be greater than zero."
        )

    # ==================================================
    # 2. LOAD JOURNEY DATA
    # ==================================================

    df = pd.read_csv(journeys_file)

    required_columns = [
        date_column,
        duration_column,
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
            f"Missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df[date_column] = pd.to_datetime(
        df[date_column],
        dayfirst=True,
        errors="coerce"
    )

    coordinate_columns = [
        "latitude_start",
        "longitude_start",
        "latitude_end",
        "longitude_end"
    ]

    for col in coordinate_columns + [duration_column]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # ==================================================
    # 3. CALCULATE FULL NETWORK EXTENT
    # ==================================================

    start_coords = df[
        ["longitude_start", "latitude_start"]
    ].rename(
        columns={
            "longitude_start": "longitude",
            "latitude_start": "latitude"
        }
    )

    end_coords = df[
        ["longitude_end", "latitude_end"]
    ].rename(
        columns={
            "longitude_end": "longitude",
            "latitude_end": "latitude"
        }
    )

    stations = pd.concat(
        [start_coords, end_coords],
        ignore_index=True
    )

    stations = stations.dropna().drop_duplicates()

    stations = stations[
        stations["longitude"].between(-180, 180)
        & stations["latitude"].between(-90, 90)
    ]

    if stations.empty:
        raise ValueError(
            "No valid station coordinates found."
        )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:27700",
        always_xy=True
    )

    station_x, station_y = transformer.transform(
        stations["longitude"].to_numpy(),
        stations["latitude"].to_numpy()
    )

    station_x = np.asarray(station_x)
    station_y = np.asarray(station_y)

    xmin = station_x.min() - padding
    xmax = station_x.max() + padding
    ymin = station_y.min() - padding
    ymax = station_y.max() + padding

    # Same geographical extent for both maps.

    target_ratio = 1.25

    width = xmax - xmin
    height = ymax - ymin

    centre_x = (xmin + xmax) / 2
    centre_y = (ymin + ymax) / 2

    if width / height < target_ratio:
        width = height * target_ratio
    else:
        height = width / target_ratio

    xmin = centre_x - width / 2
    xmax = centre_x + width / 2

    ymin = centre_y - height / 2
    ymax = centre_y + height / 2

    map_bbox = (xmin, ymin, xmax, ymax)

    # ==================================================
    # 4. PREPARE JOURNEYS FOR BOTH DATES
    # ==================================================

    comparison_data = {}

    for label, selected_date in [
        ("left", left_date),
        ("right", right_date)
    ]:

        daily_df = df[
            df[date_column].dt.normalize() == selected_date
        ].copy()

        daily_df = daily_df.dropna(
            subset=coordinate_columns + [duration_column]
        )

        daily_df = daily_df[
            (daily_df[duration_column] > 0)
            & daily_df["longitude_start"].between(-180, 180)
            & daily_df["longitude_end"].between(-180, 180)
            & daily_df["latitude_start"].between(-90, 90)
            & daily_df["latitude_end"].between(-90, 90)
        ].copy()

        if daily_df.empty:
            raise ValueError(
                f"No valid journeys found for "
                f"{selected_date.date()}."
            )

        daily_df["duration_minutes"] = (
            daily_df[duration_column]
            / conversion[duration_unit]
        )

        # Project endpoints into British National Grid.

        x1, y1 = transformer.transform(
            daily_df["longitude_start"].to_numpy(),
            daily_df["latitude_start"].to_numpy()
        )

        x2, y2 = transformer.transform(
            daily_df["longitude_end"].to_numpy(),
            daily_df["latitude_end"].to_numpy()
        )

        daily_df["x1"] = x1
        daily_df["y1"] = y1
        daily_df["x2"] = x2
        daily_df["y2"] = y2

        comparison_data[label] = daily_df

    # ==================================================
    # 5. LOAD OS OPENMAP BASEMAP ONCE
    # ==================================================

    gpkg_files = list(
        Path(map_folder).rglob("*.gpkg")
    )

    if not gpkg_files:
        raise FileNotFoundError(
            f"No GeoPackage files found in '{map_folder}'."
        )

    # Colour, drawing order, line width.
    layer_styles = {
        "surface":  ("#E7EEE5", 1, 0.3),
        "wood":     ("#DDE8D5", 2, 0.3),
        "building": ("#E5E5E5", 3, 0.2),
        "road":     ("#FFFFFF", 4, 0.5),
        "rail":     ("#AAAAAA", 5, 0.4),
        "water":    ("#B8D7E8", 6, 0.3)
    }

    extent_polygon = gpd.GeoDataFrame(
        geometry=[
            box(xmin, ymin, xmax, ymax)
        ],
        crs="EPSG:27700"
    )

    basemap_layers = []

    for gpkg_file in gpkg_files:

        try:
            available_layers = gpd.list_layers(
                gpkg_file
            )

        except Exception as e:
            print(
                f"Could not inspect {gpkg_file}: {e}"
            )
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

            colour, zorder, linewidth = matched_style

            try:
                # Inspect the source CRS.
                layer_info = gpd.read_file(
                    gpkg_file,
                    layer=layer_name,
                    rows=1
                )

                if layer_info.crs is None:
                    continue

                # Convert map extent to source CRS.
                source_bbox = tuple(
                    extent_polygon
                    .to_crs(layer_info.crs)
                    .total_bounds
                )

                # Read features within the map extent.
                layer = gpd.read_file(
                    gpkg_file,
                    layer=layer_name,
                    bbox=source_bbox
                )

                if layer.empty:
                    continue

                layer = layer.to_crs("EPSG:27700")

                layer = layer[
                    layer.geometry.geom_type.isin([
                        "Polygon",
                        "MultiPolygon",
                        "LineString",
                        "MultiLineString"
                    ])
                ]

                if layer.empty:
                    continue

                basemap_layers.append(
                    (layer, colour, zorder, linewidth)
                )

                print(
                    f"Loaded {layer_name}: "
                    f"{len(layer):,} features"
                )

            except Exception as e:
                print(
                    f"Could not load {layer_name}: {e}"
                )

    if not basemap_layers:
        raise RuntimeError(
            "No OS OpenMap basemap layers were loaded.\n"
            "Check the GeoPackage folder and layer names."
        )

    # ==================================================
    # 6. CREATE COMPACT FIGURE
    # ==================================================

    # Fixed A4 landscape width.
    figure_width = 11.69

    # Horizontal margins.
    left_margin = 0.02
    right_margin = 0.02
    panel_gap = 0.02

    usable_width = (
        figure_width
        * (1 - left_margin - right_margin - panel_gap)
    )

    panel_width = usable_width / 2

    # Calculate height from map geography.
    map_ratio = (xmax - xmin) / (ymax - ymin)

    map_height = panel_width / map_ratio

    # Physical space reserved for labels.
    title_space = 0.38 if show_titles else 0.08
    colourbar_space = 0.9 if show_colorbar else 0.12

    figure_height = (
        map_height
        + title_space
        + colourbar_space
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(figure_width, figure_height)
    )

    fig.patch.set_facecolor("white")

    map_bottom = colourbar_space / figure_height

    map_top = (
        colourbar_space + map_height
    ) / figure_height

    fig.subplots_adjust(
        left=left_margin,
        right=1 - right_margin,
        bottom=map_bottom,
        top=map_top,
        wspace=panel_gap
    )

    cmap = plt.get_cmap("YlOrRd")

    norm = mcolors.Normalize(
        vmin=0,
        vmax=max_duration_minutes,
        clip=True
    )

    # ==================================================
    # 7. HELPER FUNCTION: DRAW JOURNEY ARROWS
    # ==================================================

    def draw_journeys(ax, daily_df):

        x1 = daily_df["x1"].to_numpy()
        y1 = daily_df["y1"].to_numpy()
        x2 = daily_df["x2"].to_numpy()
        y2 = daily_df["y2"].to_numpy()

        durations = (
            daily_df["duration_minutes"].to_numpy()
        )

        colours = cmap(norm(durations))

        # ----------------------------------------------
        # Journey lines
        # ----------------------------------------------

        segments = np.stack(
            [
                np.column_stack([x1, y1]),
                np.column_stack([x2, y2])
            ],
            axis=1
        )

        line_collection = LineCollection(
            segments,
            colors=colours,
            linewidths=line_width,
            alpha=line_alpha,
            zorder=10,
            rasterized=True
        )

        ax.add_collection(line_collection)

        # ----------------------------------------------
        # Directional arrowheads
        # ----------------------------------------------

        dx = x2 - x1
        dy = y2 - y1

        lengths = np.hypot(dx, dy)

        valid = lengths > 0

        dx = dx[valid]
        dy = dy[valid]
        lengths = lengths[valid]

        x2_valid = x2[valid]
        y2_valid = y2[valid]

        arrow_colours = colours[valid]

        ux = dx / lengths
        uy = dy / lengths

        px = -uy
        py = ux

        sizes = np.minimum(
            arrow_size,
            lengths * 0.35
        )

        tip_x = x2_valid - ux * sizes * 0.2
        tip_y = y2_valid - uy * sizes * 0.2

        base_x = tip_x - ux * sizes
        base_y = tip_y - uy * sizes

        half_width = sizes * 0.35

        triangles = np.stack(
            [
                np.column_stack([
                    tip_x,
                    tip_y
                ]),
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

        arrow_collection = PolyCollection(
            triangles,
            facecolors=arrow_colours,
            edgecolors="none",
            alpha=line_alpha,
            zorder=11,
            rasterized=True
        )

        ax.add_collection(arrow_collection)

    # ==================================================
    # 8. DRAW BOTH MAPS
    # ==================================================

    for ax, label, selected_date in zip(
        axes,
        ["left", "right"],
        [left_date, right_date]
    ):

        ax.set_facecolor("#F8F8F6")

        # ----------------------------------------------
        # OS OpenMap basemap
        # ----------------------------------------------

        for layer, colour, zorder, linewidth in basemap_layers:

            layer.plot(
                ax=ax,
                color=colour,
                linewidth=linewidth,
                edgecolor=colour,
                zorder=zorder
            )

        # ----------------------------------------------
        # Optional docking stations
        # ----------------------------------------------

        if show_stations:

            ax.scatter(
                station_x,
                station_y,
                s=2,
                color="#777777",
                alpha=0.35,
                linewidths=0,
                zorder=8,
                rasterized=True
            )

        # ----------------------------------------------
        # Journey arrows
        # ----------------------------------------------

        draw_journeys(
            ax,
            comparison_data[label]
        )

        # ----------------------------------------------
        # Identical map extent
        # ----------------------------------------------

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_aspect("equal")
        ax.set_axis_off()

        if show_titles:

            ax.set_title(
                selected_date.strftime("%d %B %Y"),
                fontsize=13,
                fontweight="bold",
                color="#222222",
                pad=8
            )

    # ==================================================
    # 9. SHARED COLOURBAR
    # ==================================================

    if show_colorbar:

        sm = plt.cm.ScalarMappable(
            norm=norm,
            cmap=cmap
        )

        sm.set_array([])

        colourbar_height = 0.12 / figure_height
        colourbar_gap = 0.20 / figure_height

        colourbar_bottom = (
            map_bottom
            - colourbar_gap
            - colourbar_height
        )

        cbar_ax = fig.add_axes([
            0.28,
            colourbar_bottom,
            0.44,
            colourbar_height
        ])

        cbar = fig.colorbar(
            sm,
            cax=cbar_ax,
            orientation="horizontal",
            extend="max"
        )

        cbar.set_label(
            "Journey duration (minutes)",
            fontsize=12,
            labelpad=6
        )

        cbar.ax.tick_params(
            labelsize=10,
            length=3,
            width=0.8,
            pad=4
        )

        cbar.outline.set_linewidth(0.6)

    # ==================================================
    # 10. EXPORT
    # ==================================================

    if output_name is None:
        output_name = (
            f"journey_comparison_"
            f"{left_date:%Y-%m-%d}_"
            f"{right_date:%Y-%m-%d}"
        )

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

    # ==================================================
    # 11. PRINT SUMMARY
    # ==================================================

    for label, selected_date in [
        ("left", left_date),
        ("right", right_date)
    ]:

        daily_df = comparison_data[label]

        print(
            f"\n{label.capitalize()} map: "
            f"{selected_date.date()}"
        )

        print(
            f"Journeys plotted: "
            f"{len(daily_df):,}"
        )

        print(
            f"Median journey duration: "
            f"{daily_df['duration_minutes'].median():.1f} minutes"
        )

    return fig, axes, comparison_data

fig, axes, comparison_data = plot_comparative_journey_arrows(
    date_left="2025-04-06",
    date_right="2025-04-27",
    journeys_file="TestData.csv",
    map_folder="opmplc_gpkg_gb",
    date_column="Start date",
    duration_column="Total duration (ms)",
    duration_unit="ms",
    padding=600,
    max_duration_minutes=60,
    line_alpha=0.15,
    line_width=0.7,
    arrow_size=65,
    show_colorbar=True,
    show_stations=False,
    show_titles=True
)