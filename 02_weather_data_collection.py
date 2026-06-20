"""
Launch Weather Data Collection
================================

For every rocket launch in `launches_2014_2025.csv`, queries the
Open-Meteo historical weather API (https://open-meteo.com/) for daily
weather conditions at the launch pad's coordinates, covering the launch
window's start/end dates.

Open-Meteo has no documented hard rate limit for this endpoint, but
intermittent timeouts and connection errors are common, so requests are
wrapped in a retry session and progress is saved after every launch.

Output
------
- launch_weather.json : raw weather API responses, one entry per launch
- launch_dataset.csv  : launches merged with their corresponding weather,
                        ready for the data cleaning step (03_data_cleaning.py)
"""

import json
import time

import numpy as np
import pandas as pd
import requests
from retry_requests import retry

from data_utils import save_json, load_json

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LAUNCHES_CSV_PATH = "launches_2014_2025.csv"
WEATHER_RAW_JSON_PATH = "launch_weather.json"
MERGED_OUTPUT_CSV_PATH = "launch_dataset.csv"
OPEN_METEO_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Daily weather variables to request from Open-Meteo for each launch
DAILY_WEATHER_VARS = [
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours",
    "cloud_cover_mean", "cloud_cover_max", "cloud_cover_min",
    "temperature_2m_min", "temperature_2m_max", "weather_code",
    "relative_humidity_2m_mean", "relative_humidity_2m_max", "relative_humidity_2m_min",
    "surface_pressure_mean", "surface_pressure_max", "surface_pressure_min",
    "winddirection_10m_dominant", "wind_gusts_10m_mean", "wind_speed_10m_mean",
    "wind_speed_10m_min", "wind_gusts_10m_min", "snowfall_water_equivalent_sum",
    "pressure_msl_mean", "pressure_msl_max", "pressure_msl_min", "temperature_2m_mean",
]

RATE_LIMIT_SLEEP_SECONDS = 60
RETRY_BACKOFF_FACTOR = 0.5
MAX_RETRIES = 5


def build_weather_query_subset(launches_df):
    """
    Extract the columns needed to query the weather API for each launch.

    Parameters
    ----------
    launches_df : pandas.DataFrame
        Flattened launch data produced by 01_launch_data_collection.py.

    Returns
    -------
    pandas.DataFrame
        Subset with launch id, pad coordinates, and launch window dates.
    """
    return launches_df[["id", "pad.latitude", "pad.longitude", "window_start_date", "window_end_date"]].copy()


def fetch_weather_for_launches(launch_subset, daily_vars, base_url, raw_json_path):
    """
    Query Open-Meteo for daily weather at each launch's pad coordinates
    and launch window, saving progress to `raw_json_path` after each one.

    Parameters
    ----------
    launch_subset : pandas.DataFrame
        Output of `build_weather_query_subset`; must contain `id`,
        `pad.latitude`, `pad.longitude`, `window_start_date`, `window_end_date`.
    daily_vars : list of str
        Open-Meteo daily variable names to request.
    base_url : str
        Open-Meteo archive endpoint URL.
    raw_json_path : str
        Path to persist results to after every successful request.

    Returns
    -------
    list of dict
        One entry per launch: {"launch_id": ..., "weather": <api response>}.
    """
    session = retry(requests.Session(), retries=MAX_RETRIES, backoff_factor=RETRY_BACKOFF_FACTOR)
    weather_records = []

    for idx, row in launch_subset.iterrows():
        params = {
            "latitude": row["pad.latitude"],
            "longitude": row["pad.longitude"],
            "start_date": row["window_start_date"],
            "end_date": row["window_end_date"],
            "daily": daily_vars,
            "timezone": "UTC",
        }

        while True:
            try:
                response = session.get(base_url, params=params, timeout=30)
            except Exception as exc:
                print(f"Request failed at row {idx}: {exc}")
                time.sleep(10)
                continue

            # Handle rate limit error (429)
            if response.status_code == 429:
                print(f"Rate limit hit. Sleeping for {RATE_LIMIT_SLEEP_SECONDS}s before retrying...")
                time.sleep(RATE_LIMIT_SLEEP_SECONDS)
                continue

            # Handle all other errors
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                break

            try:
                weather_payload = response.json()
            except json.JSONDecodeError:
                print("JSON decode error, retrying current row.")
                print(response.text)
                time.sleep(10)
                continue

            # launch_id is kept as a primary key to merge back onto the
            # launch dataset later.
            weather_records.append({"launch_id": row["id"], "weather": weather_payload})

            save_json(weather_records, raw_json_path)
            print(f"Total weather collected: {len(weather_records)}")
            break

    print(f"Collection complete! Total weather collected: {len(weather_records)}")
    return weather_records


def build_weather_dataframe(weather_records):
    """
    Flatten raw weather JSON into a DataFrame and collapse multi-day
    values into a single number per launch.

    Open-Meteo returns one value per calendar day covered by the launch
    window. Most launches fall within a single day, but windows that span
    midnight produce a list of 2+ values per variable; these are averaged
    so every launch ends up with one scalar value per weather variable.

    Parameters
    ----------
    weather_records : list of dict
        Output of `fetch_weather_for_launches`.

    Returns
    -------
    pandas.DataFrame
        One row per launch, with `weather.daily.*` columns coerced to
        numeric and averaged across days.
    """
    weather_df = pd.json_normalize(weather_records)

    daily_weather_cols = [c for c in weather_df.columns if c.startswith("weather.daily.")]

    for col in daily_weather_cols:
        weather_df[col] = weather_df[col].apply(pd.to_numeric, errors="coerce")
        weather_df[col] = weather_df[col].apply(np.average)

    return weather_df


def merge_launches_with_weather(launches_df, weather_df):
    """
    Left-join launches with their corresponding weather records.

    Parameters
    ----------
    launches_df : pandas.DataFrame
        Flattened launch data.
    weather_df : pandas.DataFrame
        Output of `build_weather_dataframe`.

    Returns
    -------
    pandas.DataFrame
        Launches enriched with weather columns. Every launch should have
        a matching weather record, since weather was collected for every
        row in `launches_df`.
    """
    return launches_df.merge(weather_df, how="left", left_on="id", right_on="launch_id")


def main():
    launches_df = pd.read_csv(LAUNCHES_CSV_PATH)

    weather_query_subset = build_weather_query_subset(launches_df)
    fetch_weather_for_launches(
        weather_query_subset, DAILY_WEATHER_VARS, OPEN_METEO_BASE_URL, WEATHER_RAW_JSON_PATH
    )

    # Reload from disk so the merge step always reflects exactly what was saved.
    weather_records = load_json(WEATHER_RAW_JSON_PATH)
    weather_df = build_weather_dataframe(weather_records)

    print(len(weather_df))
    print(weather_df.columns)
    print(weather_df.dtypes.to_string())

    launch_dataset = merge_launches_with_weather(launches_df, weather_df)

    print(f"Total rows after merge: {len(launch_dataset)}")
    launch_dataset.to_csv(MERGED_OUTPUT_CSV_PATH, header=True, index=False)
    print(f"Saved merged dataset to {MERGED_OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
