from pydantic import BaseModel, Field


class LogSearchInput(BaseModel):
    """Inputs for log search tool."""

    start_time: str = Field(
        ..., description="Start time in format 'YYYY-MM-DD HH:MM:SS'"
    )
    end_time: str = Field(..., description="End time in format 'YYYY-MM-DD HH:MM:SS'")
    query: str = Field(
        ..., description="Loki query string (e.g. '{job=~\"default/auth\"}')"
    )
    limit: int = Field(5000, description="Maximum number of logs to return")


class ModuleSelectionInput(BaseModel):
    """Inputs for module selection tool."""

    walkthrough: str = Field(..., description="Description of the codebase/system")
    file_tree: str = Field(..., description="File tree structure of the codebase")
    issue: str = Field(..., description="Description of the production issue")
