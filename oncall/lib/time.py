from datetime import datetime, timezone


def to_unix_nano(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9)
