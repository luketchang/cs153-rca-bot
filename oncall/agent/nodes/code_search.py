import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Dict, List, TypedDict, operator

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from oncall.agent.models import TaskComplete
from oncall.agent.nodes.utils import load_file_tree
from oncall.code.tool import CodeSearchInput, CodeSearchTool
from oncall.lib.utils import get_llm

PROMPT_PREFIX = """
You are an expert AI assistant that assists engineers debugging production issues. You specifically help by finding sections of the code relevant to the issue and a request of what type of code is needed.
"""

PROMPT_TEMPLATE = f"""
Given a request for the type of code needed for the investigation, paths to all directories/files within the codebase, previously fetched context during your exploration, your task is to fetch files relevant to the request. You will do so by outputting your intermediate reasoning then outputting EITHER a CodeSearchInput to read from another directory OR a TaskComplete to indicate that you have completed the request.

Guidelines:
- Only gather as much context as the request tells you to, DO NOT do any more than that. Do not include in your reasoning that you should gather more context than what the request suggests.
- When you output a CodeSearchInput, you will get back all files in that directory recursively.
- Refer to the provided codebase overview for guidance on which areas to inspect.
- Utilize the codebase path to construct absolute paths for the ripgrep tool mappings—ABSOLUTE PATHS MUST BE USED.
- Look at context you previously gathered and see which directories you have already searched for. DO NOT issue a search for a child directory or the same directory twice.
- AVOID searching the same files multiple times, it is a waste of the context window.
- Keep your searches narrow in scope, to minimize the amount of unnecessary context you gather.
- Stay targeted with your code search but do not just end up searching the whole repository. Only search in the directories of services that are directly related to your issue and the request, do not stray too far from the issue and request.
  - For example, if the request tells me to search for code in the payments service, I SHOULD NOT search for code in the auth or tickets services.

- DO NOT use XML tags
- If you failed to fetch any context DO NOT give up and output TaskComplete, keep adjusting your query until you get some code

CURRENT TIME (UTC): {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")}

CODEBASE PATH: {{repo_path}}

CODEBASE OVERVIEW:
{{codebase_overview}}

ALL FILEPATHS:
{{file_tree}}

ALREADY EXPLORED DIRECTORIES (DO NOT SEARCH ANY OF THESE OR THEIR CHILDREN DIRS):
{{visited_directories}}

Use the below history of context you already gathered to inform what steps you will take next. DO NOT visit the same directory or its children twice (look at the above list). 

ISSUE: 
{{issue}}

REQUEST (WHAT CODE TO SEARCH FOR):
{{request}}

CONTEXT YOU PREVIOUSLY GATHERED:
{{chat_history}}
"""


class CodeSearchOutput(BaseModel):
    reasoning: str = Field(
        ...,
        description="Intermediate reasoning explaining why you will search for the code you search for or why you are at TaskComplete (request was fullfilled with correct code context). Ask yourself if you are straying too far away from the initial request and if you are, you should output TaskComplete below.",
    )
    output: CodeSearchInput | TaskComplete


class CodeSearch:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            prefix=PROMPT_PREFIX,
            template=PROMPT_TEMPLATE,
            input_variables=[
                "issue",
                "request",
                "repo_path",
                "codebase_overview",
                "file_tree",
                "visited_directories",
                "chat_history",
            ],
        )
        self.chain = self.prompt | self.llm.with_structured_output(
            CodeSearchOutput, method="function_calling"
        )

    def invoke(
        self,
        issue: str,
        request: str,
        repo_path: str,
        codebase_overview: str,
        file_tree: str,
        visited_directories: List[str],
        chat_history: List[str],
    ) -> CodeSearchOutput:
        formatted_chat_history = "\n\n".join(
            [f"{i + 1}. {entry}" for i, entry in enumerate(chat_history)]
        )

        response = self.chain.invoke(
            {
                "issue": issue,
                "request": request,
                "repo_path": repo_path,
                "codebase_overview": codebase_overview,
                "file_tree": file_tree,
                "visited_directories": visited_directories,
                "chat_history": formatted_chat_history,
            }
        )
        return response


class CodeSearchAgentState(TypedDict):
    first_pass: bool
    issue: str
    request: str
    repo_path: str
    codebase_overview: str
    file_tree: str
    visited_directories: Annotated[List[str], operator.add]
    chat_history: Annotated[List[str], operator.add]


class CodeSearchAgent(Runnable):
    def __init__(self, llm):
        self.llm = llm
        self.compiled_graph = self.build_graph()

    def code_search(
        self,
        state: CodeSearchAgentState,
    ):
        searcher = CodeSearch(self.llm)
        response = searcher.invoke(
            issue=state["issue"],
            request=state["request"],
            repo_path=state["repo_path"],
            codebase_overview=state["codebase_overview"],
            file_tree=state["file_tree"],
            visited_directories=state["visited_directories"],
            chat_history=state["chat_history"],
        )

        output = response.output
        if isinstance(output, TaskComplete):
            print("Reasoning: ", response.reasoning)
            print("-- Task Complete --")
            return Command(goto=END)
        elif isinstance(output, CodeSearchInput):
            code_ctx = CodeSearchTool()._run([output.directory])
            code_ctx = "EMPTY" if not code_ctx else code_ctx
            print("Reasoning: ", response.reasoning)
            print(f"Read directory: {output.directory}")
            print("\n\n")

            chat_message = f"Searched for: {output.directory}\n\n Results:\n {code_ctx}"
            return Command(
                update={
                    "chat_history": [chat_message],
                    "visited_directories": [output.directory],
                },
                goto="code_search",
            )

    def build_graph(self) -> CompiledStateGraph:
        node_map = {
            "code_search": self.code_search,
        }

        graph = StateGraph(CodeSearchAgentState)
        graph.add_node("code_search", node_map["code_search"])
        graph.add_edge(START, "code_search")

        return graph.compile()

    def invoke(self, input: Dict) -> Dict:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Invokes the compiled state graph with the input data and returns the response.
        Handles recursive calls with a configurable recursion limit.
        """
        config = {"recursion_limit": 50}
        state = self.compiled_graph.invoke(input, config=config)

        return {
            "visited_directories": state["visited_directories"],
            "chat_history": state["chat_history"],
        }


def main():
    repo_path = Path("/Users/luketchang/code/Microservices_Udemy/ticketing")
    overview_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "codebase_overviews",
        "typescript-microservices.txt",
    )
    with open(overview_path, "r") as f:
        overview = f.read()
    file_tree = load_file_tree(repo_path)

    llm = get_llm("o3-mini")

    state = {
        "first_pass": True,
        "issue": "User is reporting that they have purchased tickets, payment went through, but ticket later showed as cancelled. This was roughly 2025-03-03 around 3:25am UTC. User's email was something like hello@gmail.com.",
        "request": "Search for any code in the payments service related to the issue.",
        "repo_path": str(repo_path),
        "codebase_overview": overview,
        "file_tree": file_tree,
        "visited_directories": [],
        "chat_history": [],
    }

    agent = CodeSearchAgent(llm)
    response = agent.invoke(state)
    print(response)


if __name__ == "__main__":
    main()
