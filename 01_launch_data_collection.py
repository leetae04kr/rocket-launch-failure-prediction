"""
Rocket Launch Data Collection
==============================

Collects historical orbital launch records from the Launch Library 2
(LL2) API (https://thespacedevs.com/llapi) for the period 2014-01-01 to
2025-12-31, and stores them as both a raw JSON file and a flattened CSV.

The LL2 API is rate-limited to 15 requests/hour, so this script saves its
progress after every page and can safely be re-run to resume a previously
interrupted collection instead of starting over.

Output
------
- launches_2014_2025.json : raw paginated API responses, concatenated
- launches_2014_2025.csv  : flattened launch records, ready for the next
                            pipeline step (02_weather_data_collection.py)
"""

import os
import time

import pandas as pd
import requests

from data_utils import save_json, load_json

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_JSON_PATH = "launches_2014_2025.json"
OUTPUT_CSV_PATH = "launches_2014_2025.csv"
BASE_URL = "https://ll.thespacedevs.com/2.2.0/launch/"

WINDOW_START_GTE = "2014-01-01T00:00:00Z"
WINDOW_START_LTE = "2025-12-31T23:59:59Z"

REQUEST_PAGE_SIZE = 100          # Results per page
RATE_LIMIT_SLEEP_SECONDS = 300   # Wait time after hitting a 429
REQUEST_INTERVAL_SECONDS = 250   # Wait between successful requests (15/hour cap)


def fetch_all_launches(base_url, start_date, end_date, raw_json_path):
    """
    Page through the LL2 launch endpoint and collect every launch record
    within [start_date, end_date], saving progress after each page.

    If `raw_json_path` already contains previously collected launches,
    collection resumes from where it left off (via the `offset` param)
    instead of starting over, since the API only allows 15 requests/hour.

    Parameters
    ----------
    base_url : str
        LL2 launch endpoint URL.
    start_date, end_date : str
        ISO-8601 datetime strings bounding `window_start`.
    raw_json_path : str
        Path used both to resume previous progress and to persist new
        results after every page.

    Returns
    -------
    list of dict
        All launch records collected for the requested window.
    """
    if os.path.exists(raw_json_path):
        launches = load_json(raw_json_path)
        print(f"Resuming with {len(launches)} launches already saved.")
    else:
        launches = []

    params = {
        "window_start__gte": start_date,
        "window_start__lte": end_date,
        "limit": REQUEST_PAGE_SIZE,
        "ordering": "window_start",   # Sort ascending so resuming via offset is reliable
        "offset": len(launches),      # Skip rows already collected
    }

    next_url = base_url
    while next_url:
        print(f"Requesting: {next_url}")
        response = requests.get(next_url, params=params)

        # Handle rate limit error (429)
        if response.status_code == 429:
            print(f"Rate limit hit. Sleeping for {RATE_LIMIT_SLEEP_SECONDS}s before retrying...")
            time.sleep(RATE_LIMIT_SLEEP_SECONDS)
            continue

        # Handle all other errors
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            break

        payload = response.json()
        launches.extend(payload["results"])

        save_json(launches, raw_json_path)
        print(f"Saved! Total launches collected: {len(launches)}")

        # Move to next page; `next` already encodes all filters, so params
        # are only needed for the very first request.
        next_url = payload.get("next")
        params = None

        print(f"Waiting {REQUEST_INTERVAL_SECONDS}s due to the 15 requests/hour limit...")
        time.sleep(REQUEST_INTERVAL_SECONDS)

    print(f"Extraction complete. Total launches collected: {len(launches)}")
    return launches


def build_launch_dataframe(launches):
    """
    Flatten raw nested launch JSON records into a tabular DataFrame and
    convert the columns needed downstream to numeric/datetime types.

    Parameters
    ----------
    launches : list of dict
        Raw launch records as returned by the LL2 API.

    Returns
    -------
    pandas.DataFrame
        Flattened launch data with `pad.latitude`, `pad.longitude`,
        `window_start` and `window_end` cast to proper dtypes, plus
        derived `window_start_date` / `window_end_date` string columns
        used to query the weather API in the next pipeline step.
    """
    launches_df = pd.json_normalize(launches)

    # Needed as numeric/datetime to collect weather data in the next step.
    launches_df["pad.latitude"] = pd.to_numeric(launches_df["pad.latitude"], errors="coerce")
    launches_df["pad.longitude"] = pd.to_numeric(launches_df["pad.longitude"], errors="coerce")

    launches_df["window_start"] = pd.to_datetime(launches_df["window_start"], errors="coerce")
    launches_df["window_end"] = pd.to_datetime(launches_df["window_end"], errors="coerce")

    launches_df["window_start_date"] = launches_df["window_start"].dt.strftime("%Y-%m-%d")
    launches_df["window_end_date"] = launches_df["window_end"].dt.strftime("%Y-%m-%d")

    return launches_df


def main():
    launches = fetch_all_launches(BASE_URL, WINDOW_START_GTE, WINDOW_START_LTE, RAW_JSON_PATH)
    launches_df = build_launch_dataframe(launches)

    print(launches_df.dtypes.to_string())

    # Sanity checks: date coverage and class balance of the launch outcome
    print(launches_df["window_start"].min())
    print(launches_df["window_start"].max())
    print(launches_df["status.abbrev"].value_counts())
    # Roughly 1839 successes vs. 105 failures/partial failures: the classes
    # are extremely imbalanced, which matters for the modelling step later on.

    launches_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved {len(launches_df)} launches to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
