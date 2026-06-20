"""
Shared helpers for the data collection scripts (01 and 02).

Both the launch data collection and weather data collection scripts save
their progress to a JSON file after every request, so that an interrupted
run (rate limit, crash, network error) can be resumed instead of starting
over. This module holds that shared save/load logic so it only needs to
be written and tested once.
"""

import json


def save_json(data, file_path):
    """
    Write `data` to `file_path` as pretty-printed JSON.

    Called after every request during data collection so progress is
    never more than one request away from being saved to disk.

    Parameters
    ----------
    data : list or dict
        JSON-serializable data to persist.
    file_path : str
        Destination file path.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(file_path):
    """
    Read and parse a JSON file from disk.

    Parameters
    ----------
    file_path : str
        Path to the JSON file to read.

    Returns
    -------
    list or dict
        The parsed JSON content.
    """
    with open(file_path, "r") as f:
        return json.load(f)
