from typing import List

from pydantic import BaseModel, Field


class ServiceSelection(BaseModel):
    reason: str = Field(..., description="Explanation for including this service")
    module: str = Field(..., description="Filepath for the service")


class SelectedModules(BaseModel):
    selections: List[ServiceSelection] = Field(
        ..., description="List of services/modules selected and their reasons"
    )
