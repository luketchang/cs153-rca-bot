from datetime import datetime, timezone
from typing import List

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from oncall.agent.models import TaskComplete
from oncall.agent.nodes.models import CodeRequest, LogRequest

PROMPT_PREFIX = """
You are an expert AI assistant that assists engineers debugging production issues. You specifically review root cause analyses produced by another engineer and reason about its validity and whether or the engineer is missing key context from the codebase or logs.
"""

PROMPT_TEMPLATE = f"""
Given the issue encountered, an overview of the codebase, the codebase file tree, potential log labels, and previously gathered log and code context, and a proposed root cause analysis, your  task is to question the validity of the analysis.

Go through the below checklist to check if the analysis is too imprecise or has not considered enough thorough context. If any of the below questions reveal that you have not considered enough parts of the codebase or have not retrieved logs thoroughly enough, output a `CodeRequest` or `LogRequest` respectively.

Question Checklist (just to list a few):
- Is the analysis too vague or speculative and doesn't give a real root cause?
- Is there a precise solution in the form of a code change or is it still a vague description?
- Has the engineer narrowed in one specific angle and is not seeing the full picture?
- Might the source of the issue actually be somewhere else in the logs or code that is not in the discussed in the analysis?
- Have you considered all services that could be causing the issue or have you only looked at the one demonstrating problems?
- Do you retrieved logs show a wide enough picture of the various services or do they only show logs from the actual issue occurrence?
- Do your logs actually reveal any issues or useful context?
- Does your code match the issues revealed in the logs?

Guidelines:
- Ask yourself if the proposed analysis is only considering a small part of the system or if the root cause is actually upstream in another service. If you believe its upstream or downstream, specify which services to further investigate.
- When examining logs, make sure you have a wide enough picture and do not ONLY have logs from the actual error occurrence. You should instead have a broader picture of other services and what they were doing leading up to the logs indicating the actual issue. 
- Especially in microservices, the root cause may not be in the service that is failing, but in another service that is interacting with it. Consider other services when reasoning about what you may be missing and write down those hypotheses.
- If you need more code, output a CodeRequest detailing what types of code you still need. If you need more logs, output a LogRequest detailing what types of logs you still need.

- Only output a TaskComplete if you believe the engineer has the correct root cause analysis and your previous reasoning step supported that analysis.
- The analysis should not cite some vague root cause like "Synchronization Issues" or "Performance Issues" it must be very concrete and have an actionable and exact fix. If not, then that is a sign the analysis is missing information and needs more context.

- DO NOT use XML tags

CURRENT TIME (UTC): {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")}

CODEBASE PATH: {{repo_path}}

CODEBASE OVERVIEW:
{{codebase_overview}}

FILE TREE:
{{file_tree}}

LOG LABELS:
{{labels_map}}

CONTEXT THE ENGINEER PREVIOUSLY GATHERED:
{{chat_history}}

ISSUE:
{{issue}}

ROOT CAUSE ANALYSIS:
{{root_cause}}
"""


class ReviewerOutput(BaseModel):
    reasoning: str = Field(
        ...,
        description="Analysis questioning the root cause analysis, asking if there are holes in its reasoning, asking if there are other services that should be considered, and asking if there are other parts of the code or logs that should be explored.",
    )
    output: CodeRequest | LogRequest | TaskComplete


class Reviewer:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            prefix=PROMPT_PREFIX,
            template=PROMPT_TEMPLATE,
            input_variables=[
                "issue",
                "repo_path",
                "codebase_overview",
                "file_tree",
                "labels_map",
                "chat_history",
            ],
        )
        self.chain = self.prompt | self.llm.with_structured_output(
            ReviewerOutput, method="function_calling"
        )

    def invoke(
        self,
        issue: str,
        repo_path: str,
        codebase_overview: str,
        file_tree: str,
        labels_map: str,
        chat_history: List[str],
        root_cause: str,
    ) -> ReviewerOutput:
        formatted_chat_history = "\n\n".join(
            [f"{i + 1}. {entry}" for i, entry in enumerate(chat_history)]
        )

        response = self.chain.invoke(
            {
                "issue": issue,
                "repo_path": repo_path,
                "codebase_overview": codebase_overview,
                "file_tree": file_tree,
                "labels_map": labels_map,
                "chat_history": formatted_chat_history,
                "root_cause": root_cause,
            }
        )
        print(response)
        return response
