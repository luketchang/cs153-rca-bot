from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

from oncall.constants import LOKI_PASSWORD, LOKI_USERNAME

load_dotenv()

def get_time_range(hours_back=48):
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


if __name__ == "__main__":
    base_url = "https://logs-prod-021.grafana.net"
    # Get time range for the last 1 hour
    start_ns, end_ns = get_time_range(hours_back=48)

    labels = fetch_loki_labels(base_url, start_ns, end_ns)
    if labels is None:
        print("No labels fetched.")
        exit(1)

    print("Available labels in Loki (last hour):")
    for label in labels:
        print(f"  {label}")
        values = fetch_loki_label_values(base_url, label, start_ns, end_ns)
        if values:
            print(f"    Values for '{label}':")
            for v in values:
                print(f"      {v}")
        else:
            print(f"    (No values found for '{label}')")
