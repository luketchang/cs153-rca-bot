import logging
import traceback

from pythonjsonlogger import jsonlogger

# Configure the root logger to use JSON formatting
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove any default handlers
root_logger.handlers = []

# Create and add a stream handler with JSON formatting
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = jsonlogger.JsonFormatter("%(message)s")
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)

# Set less verbose logging for external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)

twilio_logger = logging.getLogger("twilio.http_client")
twilio_logger.setLevel(logging.WARNING)

# Module-level logger for this file (and for imports)
logger = logging.getLogger(__name__)


def traceback_log_err(e: Exception):
    formatted_traceback = traceback.format_exc().replace("\n", " | ")
    logger.error(
        "Error handling task", extra={"error": str(e), "traceback": formatted_traceback}
    )
