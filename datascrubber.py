import pandas as pd
import numpy as np
import requests

from pyproj import Geod


def clean_santander_journeys(
    input_file="MarathonData.csv",
    output_file="ExtraMarathonData.csv",
    duration_column="Total duration (ms)",
    duration_unit="ms",
    min_duration_seconds=15,
    max_duration_hours=2,
    min_speed_kmh=1,
    max_speed_kmh=40,
    remove_duplicates=True,
    bike_id_column="Bike number",
    bike_model_column="Bike model",
    bike_model_threshold=0.85
):
    """
    Clean Santander Cycles journey data.

    Operations
    ----------
    1. Download docking station coordinates from TfL.
    2. Match starting and ending stations.
    3. Remove rows containing missing values.
    4. Remove invalid journey durations.
    5. Remove journeys shorter than the minimum duration.
    6. Remove journeys longer than the maximum duration.
    7. Validate station coordinates.
    8. Calculate crow-flight journey distances.
    9. Calculate implied average speeds.
    10. Remove journeys with invalid distances or speeds.
    11. Remove journeys outside the specified speed limits.
    12. Optionally remove duplicate journeys.
    13. Identify bicycles recorded with multiple models.
    14. Correct classifications when one model accounts
        for more than the specified percentage of journeys.
    15. Print a cleaning and correction summary.
    16. Save the cleaned journey dataset.

    Notes
    -----
    Bike-model corrections are calculated using journeys
    remaining after the other cleaning operations.

    Crow-flight distance is not actual cycling distance.

    Returns
    -------
    pandas.DataFrame
        Cleaned journey dataset.
    """

    # ==================================================
    # 1. VALIDATE PARAMETERS
    # ==================================================

    conversion = {
        "ms": 60000,
        "seconds": 60,
        "minutes": 1
    }

    if duration_unit not in conversion:
        raise ValueError(
            "duration_unit must be 'ms', 'seconds', or 'minutes'."
        )

    if min_duration_seconds <= 0:
        raise ValueError(
            "min_duration_seconds must be greater than zero."
        )

    if max_duration_hours <= 0:
        raise ValueError(
            "max_duration_hours must be greater than zero."
        )

    if min_duration_seconds >= max_duration_hours * 3600:
        raise ValueError(
            "Minimum duration must be less than maximum duration."
        )

    if min_speed_kmh < 0:
        raise ValueError(
            "min_speed_kmh cannot be negative."
        )

    if max_speed_kmh <= min_speed_kmh:
        raise ValueError(
            "max_speed_kmh must be greater than min_speed_kmh."
        )

    if not 0.5 < bike_model_threshold < 1:
        raise ValueError(
            "bike_model_threshold must be between 0.5 and 1."
        )

    # ==================================================
    # 2. DOWNLOAD STATION COORDINATES
    # ==================================================

    print("Downloading station coordinates from TfL...")

    url = "https://api.tfl.gov.uk/BikePoint"

    response = requests.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    stations = pd.DataFrame([
        {
            "station_name": station["commonName"],
            "latitude": station["lat"],
            "longitude": station["lon"]
        }
        for station in data
    ])

    if stations.empty:
        raise ValueError(
            "No docking stations were returned by the TfL API."
        )

    stations["latitude"] = pd.to_numeric(
        stations["latitude"],
        errors="coerce"
    )

    stations["longitude"] = pd.to_numeric(
        stations["longitude"],
        errors="coerce"
    )

    stations = stations.dropna(
        subset=[
            "station_name",
            "latitude",
            "longitude"
        ]
    )

    # Prevent ambiguous station-name matching.

    if stations["station_name"].duplicated().any():
        raise ValueError(
            "The TfL station dataset contains duplicate "
            "station names. Matching would be ambiguous."
        )

    print(
        f"Retrieved {len(stations):,} docking stations."
    )

    # ==================================================
    # 3. LOAD JOURNEY DATA
    # ==================================================

    print("\nLoading journey data...")

    journeys = pd.read_csv(
        input_file,
        low_memory=False,
        dtype={bike_id_column: "string"}
    )

    original_count = len(journeys)

    print(
        f"Original journeys: {original_count:,}"
    )

    required_columns = [
        "Start station",
        "End station",
        duration_column,
        bike_id_column,
        bike_model_column
    ]

    missing_columns = [
        col for col in required_columns
        if col not in journeys.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {journeys.columns.tolist()}"
        )

    # Remove previously generated columns if present.

    generated_columns = [
        "latitude_start",
        "longitude_start",
        "latitude_end",
        "longitude_end",
        "duration_minutes",
        "crow_flight_distance_km",
        "average_speed_kmh",
        "original_bike_model",
        "bike_model_corrected"
    ]

    journeys = journeys.drop(
        columns=generated_columns,
        errors="ignore"
    )

    # ==================================================
    # 4. STANDARDISE BIKE IDENTIFIERS AND MODELS
    # ==================================================

    journeys[bike_id_column] = (
        journeys[bike_id_column]
        .astype("string")
        .str.strip()
    )

    journeys[bike_model_column] = (
        journeys[bike_model_column]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # Treat empty strings as missing values.

    journeys[bike_id_column] = (
        journeys[bike_id_column]
        .replace("", pd.NA)
    )

    journeys[bike_model_column] = (
        journeys[bike_model_column]
        .replace("", pd.NA)
    )

    # ==================================================
    # 5. MATCH STARTING STATION COORDINATES
    # ==================================================

    start_locations = journeys.merge(
        stations[
            [
                "station_name",
                "latitude",
                "longitude"
            ]
        ],
        left_on="Start station",
        right_on="station_name",
        how="left",
        validate="many_to_one"
    )

    start_locations = start_locations.rename(
        columns={
            "latitude": "latitude_start",
            "longitude": "longitude_start"
        }
    )

    start_locations = start_locations.drop(
        columns="station_name"
    )

    # ==================================================
    # 6. MATCH ENDING STATION COORDINATES
    # ==================================================

    journeys_with_coordinates = start_locations.merge(
        stations[
            [
                "station_name",
                "latitude",
                "longitude"
            ]
        ],
        left_on="End station",
        right_on="station_name",
        how="left",
        validate="many_to_one"
    )

    journeys_with_coordinates = (
        journeys_with_coordinates.rename(
            columns={
                "latitude": "latitude_end",
                "longitude": "longitude_end"
            }
        )
    )

    journeys_with_coordinates = (
        journeys_with_coordinates.drop(
            columns="station_name"
        )
    )

    # ==================================================
    # 7. REPORT STATION MATCHING
    # ==================================================

    start_matched = (
        journeys_with_coordinates["latitude_start"]
        .notna()
        .mean() * 100
        if original_count > 0
        else 0
    )

    end_matched = (
        journeys_with_coordinates["latitude_end"]
        .notna()
        .mean() * 100
        if original_count > 0
        else 0
    )

    print("\nStation matching results:")

    print(
        f"Starting stations matched: {start_matched:.2f}%"
    )

    print(
        f"Ending stations matched: {end_matched:.2f}%"
    )

    # ==================================================
    # 8. CONVERT NUMERIC COLUMNS
    # ==================================================

    numeric_columns = [
        duration_column,
        "latitude_start",
        "longitude_start",
        "latitude_end",
        "longitude_end"
    ]

    for col in numeric_columns:

        journeys_with_coordinates[col] = pd.to_numeric(
            journeys_with_coordinates[col],
            errors="coerce"
        )

    journeys_with_coordinates = (
        journeys_with_coordinates.replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # ==================================================
    # 9. REMOVE ROWS WITH MISSING VALUES
    # ==================================================

    before_missing = len(journeys_with_coordinates)

    journeys_with_coordinates = (
        journeys_with_coordinates.dropna().copy()
    )

    removed_missing = (
        before_missing
        - len(journeys_with_coordinates)
    )

    print(
        f"\nRows removed due to missing values: "
        f"{removed_missing:,}"
    )

    # ==================================================
    # 10. CALCULATE JOURNEY DURATION
    # ==================================================

    journeys_with_coordinates["duration_minutes"] = (
        journeys_with_coordinates[duration_column]
        / conversion[duration_unit]
    )

    before_invalid = len(journeys_with_coordinates)

    journeys_with_coordinates = (
        journeys_with_coordinates[
            journeys_with_coordinates["duration_minutes"] > 0
        ].copy()
    )

    removed_invalid = (
        before_invalid
        - len(journeys_with_coordinates)
    )

    print(
        f"Rows removed due to invalid durations: "
        f"{removed_invalid:,}"
    )

    # ==================================================
    # 11. REMOVE JOURNEYS BELOW MINIMUM DURATION
    # ==================================================

    minimum_minutes = min_duration_seconds / 60

    before_short = len(journeys_with_coordinates)

    journeys_with_coordinates = (
        journeys_with_coordinates[
            journeys_with_coordinates["duration_minutes"]
            >= minimum_minutes
        ].copy()
    )

    removed_short = (
        before_short
        - len(journeys_with_coordinates)
    )

    print(
        f"Journeys removed below {min_duration_seconds} seconds: "
        f"{removed_short:,}"
    )

    # ==================================================
    # 12. REMOVE JOURNEYS OVER MAXIMUM DURATION
    # ==================================================

    maximum_minutes = max_duration_hours * 60

    before_long = len(journeys_with_coordinates)

    journeys_with_coordinates = (
        journeys_with_coordinates[
            journeys_with_coordinates["duration_minutes"]
            <= maximum_minutes
        ].copy()
    )

    removed_long = (
        before_long
        - len(journeys_with_coordinates)
    )

    print(
        f"Journeys removed over "
        f"{max_duration_hours} hours: "
        f"{removed_long:,}"
    )

    # ==================================================
    # 13. VALIDATE COORDINATES
    # ==================================================

    before_coordinates = len(journeys_with_coordinates)

    valid_coordinates = (
        journeys_with_coordinates["latitude_start"]
        .between(-90, 90)
        &
        journeys_with_coordinates["latitude_end"]
        .between(-90, 90)
        &
        journeys_with_coordinates["longitude_start"]
        .between(-180, 180)
        &
        journeys_with_coordinates["longitude_end"]
        .between(-180, 180)
    )

    journeys_with_coordinates = (
        journeys_with_coordinates[
            valid_coordinates
        ].copy()
    )

    removed_coordinates = (
        before_coordinates
        - len(journeys_with_coordinates)
    )

    print(
        f"Rows removed due to invalid coordinates: "
        f"{removed_coordinates:,}"
    )

    # ==================================================
    # 14. CALCULATE CROW-FLIGHT DISTANCE
    # ==================================================

    print("\nCalculating crow-flight distances...")

    geod = Geod(ellps="WGS84")

    longitude_start = (
        journeys_with_coordinates["longitude_start"]
        .to_numpy(dtype=float)
    )

    latitude_start = (
        journeys_with_coordinates["latitude_start"]
        .to_numpy(dtype=float)
    )

    longitude_end = (
        journeys_with_coordinates["longitude_end"]
        .to_numpy(dtype=float)
    )

    latitude_end = (
        journeys_with_coordinates["latitude_end"]
        .to_numpy(dtype=float)
    )

    _, _, distance_metres = geod.inv(
        longitude_start,
        latitude_start,
        longitude_end,
        latitude_end
    )

    journeys_with_coordinates[
        "crow_flight_distance_km"
    ] = (
        np.asarray(distance_metres) / 1000
    )

    # ==================================================
    # 15. CALCULATE IMPLIED AVERAGE SPEED
    # ==================================================

    print("Calculating implied average speeds...")

    journey_hours = (
        journeys_with_coordinates["duration_minutes"]
        / 60
    )

    journeys_with_coordinates["average_speed_kmh"] = (
        journeys_with_coordinates["crow_flight_distance_km"]
        / journey_hours
    )

    # ==================================================
    # 16. REMOVE INVALID DISTANCE/SPEED VALUES
    # ==================================================

    before_invalid_speed = len(journeys_with_coordinates)

    valid_speed_values = (
        np.isfinite(
            journeys_with_coordinates["average_speed_kmh"]
        )
        &
        np.isfinite(
            journeys_with_coordinates["crow_flight_distance_km"]
        )
    )

    journeys_with_coordinates = (
        journeys_with_coordinates[
            valid_speed_values
        ].copy()
    )

    removed_invalid_speed = (
        before_invalid_speed
        - len(journeys_with_coordinates)
    )

    print(
        f"Rows removed due to invalid distance/speed: "
        f"{removed_invalid_speed:,}"
    )

    # ==================================================
    # 17. REMOVE JOURNEYS BELOW MINIMUM SPEED
    # ==================================================

    before_slow = len(journeys_with_coordinates)

    journeys_with_coordinates = (
        journeys_with_coordinates[
            journeys_with_coordinates["average_speed_kmh"]
            >= min_speed_kmh
        ].copy()
    )

    removed_slow = (
        before_slow
        - len(journeys_with_coordinates)
    )

    print(
        f"Journeys removed below {min_speed_kmh} km/h: "
        f"{removed_slow:,}"
    )

    # ==================================================
    # 18. REMOVE JOURNEYS ABOVE MAXIMUM SPEED
    # ==================================================

    before_fast = len(journeys_with_coordinates)

    journeys_with_coordinates = (
        journeys_with_coordinates[
            journeys_with_coordinates["average_speed_kmh"]
            <= max_speed_kmh
        ].copy()
    )

    removed_fast = (
        before_fast
        - len(journeys_with_coordinates)
    )

    print(
        f"Journeys removed above {max_speed_kmh} km/h: "
        f"{removed_fast:,}"
    )

    # ==================================================
    # 19. REMOVE DUPLICATE JOURNEYS
    # ==================================================

    removed_duplicates = 0

    if remove_duplicates:

        before_duplicates = len(journeys_with_coordinates)

        journeys_with_coordinates = (
            journeys_with_coordinates.drop_duplicates()
        )

        removed_duplicates = (
            before_duplicates
            - len(journeys_with_coordinates)
        )

        print(
            f"Duplicate rows removed: "
            f"{removed_duplicates:,}"
        )

    # ==================================================
    # 20. PRESERVE ORIGINAL BIKE MODELS
    # ==================================================

    print("\nChecking bike-model consistency...")

    journeys_with_coordinates["original_bike_model"] = (
        journeys_with_coordinates[bike_model_column].copy()
    )

    journeys_with_coordinates["bike_model_corrected"] = False

    # ==================================================
    # 21. COUNT BIKE MODELS FOR EACH BICYCLE
    # ==================================================

    bike_model_counts = (
        journeys_with_coordinates
        .groupby(
            [bike_id_column, bike_model_column]
        )
        .size()
        .reset_index(name="journey_count")
    )

    bike_totals = (
        journeys_with_coordinates
        .groupby(bike_id_column)
        .size()
        .rename("total_journeys")
        .reset_index()
    )

    bike_model_counts = bike_model_counts.merge(
        bike_totals,
        on=bike_id_column,
        how="left",
        validate="many_to_one"
    )

    bike_model_counts["model_proportion"] = (
        bike_model_counts["journey_count"]
        / bike_model_counts["total_journeys"]
    )

    # ==================================================
    # 22. IDENTIFY INCONSISTENT BICYCLES
    # ==================================================

    distinct_model_counts = (
        bike_model_counts
        .groupby(bike_id_column)[bike_model_column]
        .nunique()
    )

    inconsistent_bike_ids = distinct_model_counts[
        distinct_model_counts > 1
    ].index

    number_inconsistent = len(
        inconsistent_bike_ids
    )

    print(
        f"Bicycles with multiple recorded models: "
        f"{number_inconsistent:,}"
    )

    # ==================================================
    # 23. IDENTIFY BIKES ABOVE THE MAJORITY THRESHOLD
    # ==================================================

    majority_models = bike_model_counts[
        (
            bike_model_counts[bike_id_column]
            .isin(inconsistent_bike_ids)
        )
        &
        (
            bike_model_counts["model_proportion"]
            > bike_model_threshold
        )
    ].copy()

    # A threshold above 50% guarantees that
    # no bike can have two qualifying models.

    correction_lookup = dict(
        zip(
            majority_models[bike_id_column],
            majority_models[bike_model_column]
        )
    )

    number_correctable = len(
        correction_lookup
    )

    # ==================================================
    # 24. APPLY BIKE-MODEL CORRECTIONS
    # ==================================================

    corrected_models = (
        journeys_with_coordinates[bike_id_column]
        .map(correction_lookup)
    )

    correction_mask = (
        corrected_models.notna()
        &
        (
            journeys_with_coordinates[bike_model_column]
            != corrected_models
        )
    )

    corrected_journey_count = int(
        correction_mask.sum()
    )

    journeys_with_coordinates.loc[
        correction_mask,
        bike_model_column
    ] = corrected_models.loc[
        correction_mask
    ]

    journeys_with_coordinates.loc[
        correction_mask,
        "bike_model_corrected"
    ] = True

    # ==================================================
    # 25. PRINT BIKE-MODEL CORRECTION SUMMARY
    # ==================================================

    total_bikes = (
        journeys_with_coordinates[bike_id_column]
        .nunique()
    )

    unresolved_bikes = (
        number_inconsistent - number_correctable
    )

    print("\n" + "=" * 55)
    print("BIKE-MODEL CONSISTENCY SUMMARY")
    print("=" * 55)

    print(
        f"Total unique bicycles: "
        f"{total_bikes:,}"
    )

    print(
        f"Bicycles with inconsistent models: "
        f"{number_inconsistent:,}"
    )

    print(
        f"Bicycles corrected using majority rule: "
        f"{number_correctable:,}"
    )

    print(
        f"Bicycles remaining inconsistent: "
        f"{unresolved_bikes:,}"
    )

    print(
        f"Individual journey classifications corrected: "
        f"{corrected_journey_count:,}"
    )

    print(
        f"Majority threshold: "
        f">{bike_model_threshold * 100:.1f}%"
    )

    print("=" * 55)

    # ==================================================
    # 26. FINAL DATASET SUMMARY
    # ==================================================

    final_count = len(journeys_with_coordinates)

    removed_total = (
        original_count - final_count
    )

    retention_rate = (
        final_count / original_count * 100
        if original_count > 0
        else 0
    )

    print("\n" + "=" * 55)
    print("SANTANDER CYCLES DATA CLEANING SUMMARY")
    print("=" * 55)

    print(
        f"Original journeys:          {original_count:,}"
    )

    print(
        f"Missing values:             {removed_missing:,}"
    )

    print(
        f"Invalid durations:          {removed_invalid:,}"
    )

    print(
        f"Below {min_duration_seconds} seconds:       {removed_short:,}"
    )

    print(
        f"Over {max_duration_hours} hours:              {removed_long:,}"
    )

    print(
        f"Invalid coordinates:        {removed_coordinates:,}"
    )

    print(
        f"Invalid distance/speed:     {removed_invalid_speed:,}"
    )

    print(
        f"Below {min_speed_kmh} km/h:              {removed_slow:,}"
    )

    print(
        f"Above {max_speed_kmh} km/h:             {removed_fast:,}"
    )

    print(
        f"Duplicate journeys:         {removed_duplicates:,}"
    )

    print("-" * 55)

    print(
        f"Total rows removed:         {removed_total:,}"
    )

    print(
        f"Final journeys:             {final_count:,}"
    )

    print(
        f"Data retained:              {retention_rate:.2f}%"
    )

    print(
        f"Bike models corrected:      {corrected_journey_count:,}"
    )

    print("=" * 55)

    # ==================================================
    # 27. PRINT JOURNEY STATISTICS
    # ==================================================

    if not journeys_with_coordinates.empty:

        print("\nCleaned journey statistics:")

        print(
            "Median crow-flight distance: "
            f"{journeys_with_coordinates['crow_flight_distance_km'].median():.2f} km"
        )

        print(
            "Median implied average speed: "
            f"{journeys_with_coordinates['average_speed_kmh'].median():.2f} km/h"
        )

        print(
            "Median journey duration: "
            f"{journeys_with_coordinates['duration_minutes'].median():.2f} minutes"
        )

    # ==================================================
    # 28. SAVE CLEANED DATASET
    # ==================================================

    journeys_with_coordinates = (
        journeys_with_coordinates.reset_index(drop=True)
    )

    journeys_with_coordinates.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nCleaned dataset saved to: {output_file}"
    )

    return journeys_with_coordinates


# ======================================================
# RUN THE CLEANING FUNCTION
# ======================================================

cleaned_journeys = clean_santander_journeys(
    input_file="tfl_cycles_2025.csv",
    output_file="TestData.csv",
    duration_column="Total duration (ms)",
    duration_unit="ms",
    min_duration_seconds=15,
    max_duration_hours=2,
    min_speed_kmh=0.5,
    max_speed_kmh=30,
    remove_duplicates=True,
    bike_id_column="Bike number",
    bike_model_column="Bike model",
    bike_model_threshold=0.85
)