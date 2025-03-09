import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = "gpt-4o-2024-08-06"
MODEL_TEMPERATURE = 0.0
MODEL_SEED = 56

GRAFANA_URL = "https://logs-prod-021.grafana.net"
LOKI_USERNAME = os.getenv("LOKI_USERNAME", "MISSING LOKI_USERNAME")
LOKI_PASSWORD = os.getenv("LOKI_PASSWORD", "MISSING LOKI_PASSWORD")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "MISSING DISCORD_TOKEN")
