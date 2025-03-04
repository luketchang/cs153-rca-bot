import requests
from dotenv import load_dotenv

from oncall.constants import LOKI_PASSWORD, LOKI_USERNAME
from oncall.lib.time import to_unix_nano
from oncall.logs.utils import compress_loki_logs

# Load environment variables
load_dotenv()


def fetch_loki_logs(base_url, start, end, query, limit=5000, direction="forward"):
    url = f"{base_url}/loki/api/v1/query_range"

    params = {
        "query": query,
        "start": str(to_unix_nano(start)),
        "end": str(to_unix_nano(end)),
        "limit": limit,
        "direction": direction,
    }

    try:
        response = requests.get(url, params=params, auth=(LOKI_USERNAME, LOKI_PASSWORD))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching logs: {e}")
        return None


if __name__ == "__main__":
    base_url = "https://logs-prod-021.grafana.net"
    start_time = "2025-03-03 03:22:29"
    end_time = "2025-03-03 03:26:12"
    query = '{job=~"default/(auth|payments|orders|tickets|expiration)"}'

    logs = fetch_loki_logs(base_url, start_time, end_time, query)
    formatted_logs = compress_loki_logs(logs)

    # Print each formatted log line
    for line in formatted_logs:
        print(line)
