from pydantic import BaseModel, Field


class CodeRequest(BaseModel):
    request: str = Field(
        ...,
        description="The request for the type of code needed for the investigation. This should be a specific request for service, module, package, etc.",
    )


class LogRequest(BaseModel):
    request: str = Field(
        ...,
        description="The request for the type of logs needed for the investigation. This should be a specific request for logs related to specific events, services, etc.",
    )
