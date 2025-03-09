from pydantic import BaseModel, Field


class SupportTicket(BaseModel):
    description: str = Field(..., description="Description of the issue")
    datetime: str = Field(..., description="Datetime of the issue, must be populated")


class ChatResponse(BaseModel):
    ticket_or_followup: SupportTicket | str = Field(
        ...,
        description="Support ticket if issue has been described with a clear timestamp, otherwise a follow-up question asking for more information",
    )