from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

from oncall.constants import LOKI_PASSWORD, LOKI_USERNAME
from oncall.lib.time import to_unix_nano

load_dotenv()


def get_time_range(hours_back=24):
    now = datetime.utcnow()
    start = now - timedelta(hours=hours_back)
    start_ns = int(start.timestamp() * 1e9)
    end_ns = int(now.timestamp() * 1e9)
    return start_ns, end_ns


def fetch_loki_labels(base_url, start_ns, end_ns):
    url = f"{base_url}/loki/api/v1/labels"
    params = {"start": start_ns, "end": end_ns}
    try:
        response = requests.get(url, params=params, auth=(LOKI_USERNAME, LOKI_PASSWORD))
        response.raise_for_status()
        data = response.json()  # Expected: { "status": "success", "data": [ ... ] }
        return data.get("data", [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching labels: {e}")
        return None


def fetch_loki_label_values(base_url, label_name, start_ns, end_ns):
    url = f"{base_url}/loki/api/v1/label/{label_name}/values"
    params = {"start": start_ns, "end": end_ns}
    try:
        response = requests.get(url, params=params, auth=(LOKI_USERNAME, LOKI_PASSWORD))
        response.raise_for_status()
        data = response.json()  # Expected: { "status": "success", "data": [ ... ] }
        return data.get("data", [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching values for label '{label_name}': {e}")
        return None


def build_labels_map(base_url, start, end):
    """
    Returns a dictionary mapping each label to its list of values.
    """
    start_ns, end_ns = to_unix_nano(start), to_unix_nano(end)
    labels = fetch_loki_labels(base_url, start_ns, end_ns)
    if labels is None:
        return {}

    labels_map = {}
    for label in labels:
        values = fetch_loki_label_values(base_url, label, start_ns, end_ns)
        if values:
            labels_map[label] = values
        else:
            labels_map[label] = []
    return labels_map


if __name__ == "__main__":
    base_url = "https://logs-prod-021.grafana.net"
    labels_map = build_labels_map(base_url, hours_back=24)

    print("Labels and their values in Loki (last 24 hours):")
    for label, values in labels_map.items():
        print(f"  {label}:")
        if values:
            for v in values:
                print(f"      {v}")
        else:
            print("      (No values found)")
