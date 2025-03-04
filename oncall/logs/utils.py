import json
from typing import List


def compress_loki_logs(loki_response) -> List[str]:
    data = loki_response.get("data", {})
    result = data.get("result", [])
    formatted_logs = []

    for stream_obj in result:
        stream_labels = stream_obj.get("stream", {})
        job = stream_labels.get("job", "N/A")

        # Each 'values' item is [<ns_timestamp>, <log_line>]
        for ns_ts, raw_line in stream_obj.get("values", []):
            # Attempt to parse the log line as JSON
            try:
                parsed_line = json.loads(raw_line)
                # Extract the actual log message and the "time" field if present
                log_message = parsed_line.get("log", raw_line)
                log_time = parsed_line.get("time", None)
            except json.JSONDecodeError:
                log_message = raw_line
                log_time = None

            # Fall back to the Loki-provided nanosecond timestamp if no embedded time is found.
            final_time = log_time if log_time else ns_ts
            formatted_logs.append(f"{job} {final_time} {log_message.strip()}")

    return formatted_logs
