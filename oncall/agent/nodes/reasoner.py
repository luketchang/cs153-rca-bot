from datetime import datetime, timezone
from typing import List

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from oncall.agent.nodes.models import CodeRequest, LogRequest

PROMPT_PREFIX = """
You are an expert AI assistant that assists engineers debugging production issues. You specifically review context gathered from logs and code and question whether or not you have enough information. If you do you output a proposed root cause analysis. If not, you output a request for more logs or code.
"""

PROMPT_TEMPLATE = f"""
Given the issue encountered, an overview of the codebase, the codebase file tree, potential log labels, and previously gathered log and code context, your task is to question whether or not you have enough context. You will reason about whether or not you have enough context by outputting an intermediate `reasoning` block. Then you will output either a `CodeRequest` or `LogRequest` if you feel you need more context or a `RootCauseAnalysis` if you feel you have enough context to output a root cause analysis.

Go through the below checklist to check if the analysis is too imprecise or has not considered enough thorough context. If any of the below questions reveal that you have not considered enough parts of the codebase or have not retrieved logs thoroughly enough, output a `CodeRequest` or `LogRequest` respectively.

Question Checklist (just to list a few):
- Have you considered all services that could be causing the issue or have you only looked at the one demonstrating problems?
- Do you retrieved logs show a wide enough picture of the various services or does your context only show logs from the actual issue occurrence for a single service?
- Do your logs actually reveal any issues or useful context?
- Does your code match any issues revealed in the logs?

Guidelines:
- Especially in microservices, the root cause may not be in the service that is failing, but in another service that is interacting with it. Consider other services when reasoning about what you may be missing and write down those hypotheses.
- Only output a `RootCauseAnalysis` if you have enough context, have reasoned about a confident answer, and did not previously indicate to yourself that there are other services/hypotheses to explore.
- The root cause analysis, if you choose you are ready, should not cite some vague root cause like "Synchronization Issues" or "Performance Issues" it must be very concrete and have an actionable and exact fix. If not, then that is a sign the analysis is missing information and needs more context.
- Your root cause analysis should explicitly cite the blocks of code and where the are issues, adding inline comments to code to denote where the problem is.
- DO NOT use XML tags

CURRENT TIME (UTC): {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")}

CODEBASE PATH: {{repo_path}}

CODEBASE OVERVIEW:
{{codebase_overview}}

FILE TREE:
{{file_tree}}

LOG LABELS:
{{labels_map}}

CONTEXT YOU PREVIOUSLY GATHERED:
{{chat_history}}

ISSUE:
{{issue}}
"""


class RootCauseAnalysis(BaseModel):
    root_cause: str = Field(
        ...,
        description="The root cause of the production issue. Must be specific and walk through the sequence of events that led to the issue. Include any relevant code snippets or log lines.",
    )


class ReasonerOutput(BaseModel):
    reasoning: str = Field(
        ...,
        description="Intermediate reasoning to explain what is happening to the system given your existing context. Then enumerate the other services you may want to look into and ask if you are missing context on those services.",
    )
    output: CodeRequest | LogRequest | RootCauseAnalysis


class Reasoner:
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
            ReasonerOutput, method="function_calling"
        )

    def invoke(
        self,
        issue: str,
        repo_path: str,
        codebase_overview: str,
        file_tree: str,
        labels_map: str,
        chat_history: List[str],
    ) -> ReasonerOutput:
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
            }
        )
        print(response)
        return response
