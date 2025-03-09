from typing import Literal, Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from oncall.logs.query import fetch_loki_logs
from oncall.logs.utils import compress_loki_logs


class LogSearchInput(BaseModel):
    type: Literal["LogSearchInput"] = "LogSearchInput"
    query: str = Field(
        ...,
        description="The Loki LogQL query string to search logs, e.g. '{service_name=\"payments\"}'. Only specify at most one label/selector per query",
    )
    start: str = Field(
        ...,
        description="Start time in format 'YYYY-MM-DD HH:MM:SS'. Be generous and give +/- 15 minutes if user provided exact time.",
    )
    end: str = Field(
        ...,
        description="End time in format 'YYYY-MM-DD HH:MM:SS'. Be generous and give +/- 15 minutes if user provided exact time.",
    )
    limit: Optional[int] = Field(
        default=100,
        description="Maximum number of log lines to return, default to 500",
    )


class LogSearchTool(BaseTool):
    name: str = "loki_search"
    description: str = """
    Search Grafana Loki logs using a LogQL query within a specified time range.
    Returns compressed logs as a string with each log on a new line
    
    Guidelines
    - Keep time ranges under a few hours max to avoid timeouts or errors due to too many logs
    - Every query must have at least one label/selector, empty selectors (e.g. `{}`) are not allowed
    - Only use basic labels (e.g. service or app names) and simple keywords in the query, regexes are NOT allowed
    - Use "|=" to denote keyword searches
    """
    args_schema: type[BaseModel] = LogSearchInput
    base_url: str = "https://logs-prod-021.grafana.net"

    def _run(
        self,
        query: str,
        start: str,
        end: str,
        limit: int = 5000,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the Loki search tool"""

        # Fetch logs from Loki
        logs_response = fetch_loki_logs(
            base_url=self.base_url, start=start, end=end, query=query, limit=limit
        )

        if not logs_response:
            return "No logs found or error occurred during search."

        # Compress and format logs
        formatted_logs = compress_loki_logs(logs_response)

        # Join logs into a single string with each log on a new line
        return "\n".join(formatted_logs)
