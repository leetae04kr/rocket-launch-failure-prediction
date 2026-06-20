"""
Rocket Launch Dataset Cleaning
================================

Cleans the merged launch + weather dataset produced by
02_weather_data_collection.py: drops columns that won't be used for
modelling, and fills remaining missing values using domain-informed
assumptions (documented in `fill_missing_values` below).

Output
------
- launch_dataset_cleaned.csv : cleaned dataset, ready for feature
                                engineering and modelling
                                (04_feature_engineering_and_ML.ipynb)
"""

import pandas as pd

INPUT_CSV_PATH = "launch_dataset.csv"
OUTPUT_CSV_PATH = "launch_dataset_cleaned.csv"

# Columns that are empty, mostly missing, or simply won't be used in any
# downstream step.
UNUSED_COLUMNS = [
    "net_precision", "probability", "weather_concerns", "holdreason", "failreason",
    "hashtag", "infographic", "mission.launch_designator", "pad.agency_id",
    "pad.description", "pad.info_url", "mission", "net_precision.id", "net_precision.name",
    "net_precision.abbrev", "net_precision.description", "mission.orbit",
    "weather.daily.time", "url", "slug", "last_updated", "webcast_live", "image",
    "status.id", "status.name", "status.description", "launch_service_provider.url",
    "rocket.configuration.url", "mission.info_urls", "mission.vid_urls", "pad.url",
    "pad.wiki_url", "pad.map_url", "pad.location.url", "pad.location.map_image",
    "pad.map_image", "launch_id", "pad.location.description", "pad.location.timezone_name",
    "program", "mission.agencies", "weather.timezone", "weather.timezone_abbreviation",
    "weather.latitude", "weather.longitude", "weather.generationtime_ms",
    "weather.utc_offset_seconds", "weather.elevation",
]

# Mission metadata columns that carry no useful modelling signal once
# mission.type and mission.orbit.abbrev have been kept/imputed separately.
UNUSED_MISSION_COLUMNS = [
    "mission.id", "mission.name", "mission.description", "mission.orbit.id", "mission.orbit.name",
]


def drop_unused_columns(launch_dataset):
    """
    Drop columns that are empty, mostly missing, or not needed for later
    steps -- including the weather unit-label columns (e.g.
    `weather.daily_units.temperature_2m_max`), which only describe the
    units of their corresponding `weather.daily.*` value columns.

    Parameters
    ----------
    launch_dataset : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """
    weather_unit_columns = [c for c in launch_dataset.columns if c.startswith("weather.daily_units.")]
    return launch_dataset.drop(columns=UNUSED_COLUMNS + weather_unit_columns)


def fill_missing_values(launch_dataset):
    """
    Fill remaining missing values using domain-informed assumptions, and
    drop a few mission columns that don't carry useful signal.

    Assumptions
    -----------
    - `orbital_launch_attempt_count(_year)`: missing only for pads that
      don't perform orbital launches (`pad.orbital_launch_attempt_count`
      is 0 there), so these rockets are assumed not to be orbital and
      filled with 0.
    - `rocket.configuration.family` / `variant`: missing values are
      assumed to not belong to any known family/variant -> "Unknown".
    - `mission.type`: missing values are assumed to have no specific
      recorded purpose (e.g. confidential or test launch) -> "Unknown".
    - `mission.orbit.abbrev`: missing values are assumed to be test
      launches without a target orbit -> "Unknown". This column matters
      because orbit type affects payload mass, delta-v requirements, and
      mission risk.
    - `pad.country_code`: the single missing row corresponds to a mobile
      sea-launch platform not tied to any territory; based on current
      ownership, it is filled with "RUS".

    Parameters
    ----------
    launch_dataset : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """
    launch_dataset["orbital_launch_attempt_count"] = launch_dataset["orbital_launch_attempt_count"].fillna(0)
    launch_dataset["orbital_launch_attempt_count_year"] = launch_dataset["orbital_launch_attempt_count_year"].fillna(0)

    launch_dataset["rocket.configuration.family"] = launch_dataset["rocket.configuration.family"].fillna("Unknown")
    launch_dataset["rocket.configuration.variant"] = launch_dataset["rocket.configuration.variant"].fillna("Unknown")

    launch_dataset["mission.type"] = launch_dataset["mission.type"].fillna("Unknown")
    launch_dataset["mission.orbit.abbrev"] = launch_dataset["mission.orbit.abbrev"].fillna("Unknown")
    launch_dataset = launch_dataset.drop(columns=UNUSED_MISSION_COLUMNS)

    # Sea Launch / mobile platform: not tied to a fixed territory, and
    # currently Russian-owned, so missing country code is filled with "RUS".
    launch_dataset["pad.country_code"] = launch_dataset["pad.country_code"].fillna("RUS")

    return launch_dataset


def main():
    launch_dataset = pd.read_csv(INPUT_CSV_PATH)

    print(launch_dataset.isna().sum().to_string())
    print(launch_dataset.dtypes.to_string())

    launch_dataset = drop_unused_columns(launch_dataset)

    # The lone missing pad.country_code row, kept for visibility before it
    # gets filled -- see `fill_missing_values` for the reasoning.
    print(launch_dataset[launch_dataset["pad.country_code"].isna()])

    launch_dataset = fill_missing_values(launch_dataset)

    # Confirm every missing value has been addressed
    print(launch_dataset.isna().sum().to_string())
    print(launch_dataset.dtypes.to_string())

    launch_dataset.to_csv(OUTPUT_CSV_PATH, header=True, index=False)
    print(f"Saved cleaned dataset to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
