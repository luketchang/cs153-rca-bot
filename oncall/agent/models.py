from typing import Literal

from pydantic import BaseModel, Field

from oncall.code.tool import CodeSearchInput
from oncall.logs.tool import LogSearchInput


class TaskComplete(BaseModel):
    type: Literal["TaskComplete"] = "TaskComplete"


class AttemptComplete(BaseModel):
    type: Literal["AttemptComplete"] = "AttemptComplete"
    root_cause: str = Field(
        ...,
        description="The root cause of the production issue. Must be specific and walk through the sequence of events that led to the issue. Include any relevant code snippets or log lines.",
    )


class IntermediateReasoning(BaseModel):
    type: Literal["IntermediateReasoning"] = "IntermediateReasoning"
    reasoning: str = Field(
        ...,
        description="Intermediate reasoning on what context you have gathered so far, any issues you are encountering, and what you plan to do next. The main goal of this step is to promote further exploration. If you not gathered thorough context from BOTH code and logs, include that in next steps. If you have not explored other tangential services related to you issue (but maybe not directly affected), include that in next steps too.",
    )


class AgentAction(BaseModel):
    action: CodeSearchInput | LogSearchInput | IntermediateReasoning | AttemptComplete
    action: CodeSearchInput | LogSearchInput | IntermediateReasoning | AttemptComplete
