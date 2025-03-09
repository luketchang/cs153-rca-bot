import os
from pathlib import Path
from typing import Annotated, Dict, List, TypedDict, operator

from langchain_core.runnables.base import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from oncall.agent.models import TaskComplete
from oncall.agent.nodes.code_search import CodeSearchAgent
from oncall.agent.nodes.log_search import LogSearchAgent
from oncall.agent.nodes.reasoner import (
    CodeRequest,
    LogRequest,
    Reasoner,
    RootCauseAnalysis,
)
from oncall.agent.nodes.reviewer import Reviewer
from oncall.agent.nodes.utils import get_formatted_labels_map, load_file_tree
from oncall.lib.ops import operator_replace
from oncall.lib.utils import get_llm


class OncallAgentState(TypedDict):
    first_pass: bool
    code_request: Annotated[str, operator_replace]
    log_request: Annotated[str, operator_replace]
    issue: str
    repo_path: str
    codebase_overview: str
    file_tree: str
    visited_directories: Annotated[List[str], operator.add]
    labels_map: str
    chat_history: Annotated[List[str], operator.add]
    rca: Annotated[str, operator_replace]


class OncallAgent(Runnable):
    def __init__(self, reasoning_llm, fast_llm):
        self.reasoning_llm = reasoning_llm
        self.fast_llm = fast_llm
        self.compiled_graph = self.build_graph()

    def code_search(self, state: OncallAgentState):
        print("\n\n" + "=" * 25 + " Code Search " + "=" * 25)
        code_search_agent = CodeSearchAgent(self.fast_llm)
        response = code_search_agent.invoke(
            {
                "first_pass": state["first_pass"],
                "issue": state["issue"],
                "request": state["code_request"],
                "repo_path": state["repo_path"],
                "codebase_overview": state["codebase_overview"],
                "file_tree": state["file_tree"],
                "visited_directories": state["visited_directories"],
                "chat_history": state["chat_history"],
            }
        )

        if state["first_pass"]:
            return Command(
                update={
                    "chat_history": response["chat_history"],
                    "visited_directories": response["visited_directories"],
                    "first_pass": False,
                },
                goto="log_search",
            )
        else:
            return Command(update=response, goto="reasoner")

    def log_search(self, state: OncallAgentState):
        print("\n\n" + "=" * 25 + " Log Search " + "=" * 25)
        log_search_agent = LogSearchAgent(self.fast_llm)
        response = log_search_agent.invoke(
            {
                "first_pass": state["first_pass"],
                "issue": state["issue"],
                "request": state["log_request"],
                "labels_map": state["labels_map"],
                "chat_history": state["chat_history"],
            }
        )

        return Command(update=response, goto="reasoner")

    def reason(
        self,
        state: OncallAgentState,
    ):
        print("\n\n" + "=" * 25 + " Reasoning " + "=" * 25)
        reasoner = Reasoner(self.reasoning_llm)
        response = reasoner.invoke(
            issue=state["issue"],
            repo_path=state["repo_path"],
            codebase_overview=state["codebase_overview"],
            file_tree=state["file_tree"],
            labels_map=state["labels_map"],
            chat_history=state["chat_history"],
        )

        print(response)
        output = response.output
        if isinstance(output, RootCauseAnalysis):
            return Command(
                goto="reviewer",
                update={"rca": output.root_cause, "chat_history": [response.reasoning]},
            )
        elif isinstance(output, CodeRequest):
            return Command(
                update={
                    "chat_history": [response.reasoning],
                    "code_request": output.request,
                },
                goto="code_search",
            )
        elif isinstance(output, LogRequest):
            return Command(
                update={
                    "chat_history": [response.reasoning],
                    "log_request": output.request,
                },
                goto="log_search",
            )

    def review(self, state: OncallAgentState):
        print("\n\n" + "=" * 25 + " Review " + "=" * 25)
        reviewer = Reviewer(self.reasoning_llm)
        response = reviewer.invoke(
            issue=state["issue"],
            repo_path=state["repo_path"],
            codebase_overview=state["codebase_overview"],
            file_tree=state["file_tree"],
            labels_map=state["labels_map"],
            chat_history=state["chat_history"],
            root_cause=state["rca"],
        )

        output = response.output
        if isinstance(output, CodeRequest):
            return Command(
                update={
                    "chat_history": [response.reasoning],
                    "code_request": output.request,
                },
                goto="code_search",
            )
        elif isinstance(output, LogRequest):
            return Command(
                update={
                    "chat_history": [response.reasoning],
                    "log_request": output.request,
                },
                goto="log_search",
            )
        elif isinstance(output, TaskComplete):
            return Command(
                update={
                    "chat_history": [response.reasoning],
                },
                goto=END,
            )

    def build_graph(self) -> CompiledStateGraph:
        node_map = {
            "code_search": self.code_search,
            "log_search": self.log_search,
            "reasoner": self.reason,
            "reviewer": self.review,
        }

        graph = StateGraph(OncallAgentState)
        for node_name, node_func in node_map.items():
            graph.add_node(node_name, node_func)

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
            "chat_history": state["chat_history"],
            "rca": state["rca"],
        }


def main():
    # TODO: LLM figures out labels map for time
    labels_map = get_formatted_labels_map("2025-03-03 03:00:00", "2025-03-03 03:30:00")

    repo_path = Path("/Users/luketchang/code/Microservices_Udemy/ticketing")

    overview_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "codebase_overviews",
        "typescript-microservices.txt",
    )
    with open(overview_path, "r") as f:
        overview = f.read()

    file_tree = load_file_tree(repo_path)

    # issue = """
    # User is reporting that they have purchased tickets, payment went through, but ticket later showed as cancelled. This was roughly 2025-03-03 around 3:25am UTC. User's email was something like hello@gmail.com.
    # """

    issue = """
    Getting some "order not found" errors in the payments service which is causing crashes. One of these happened around 03-06-2025 around 4am UTC. Why is this happening?
    """

    state = {
        "first_pass": True,
        "issue": issue,
        "code_request": "Search for any code in the payments service related to the issue.",
        "log_request": "Search for any logs in the payments service related to the issue.",
        "request": "Search for any code in the payments service related to the issue.",
        "repo_path": str(repo_path),
        "codebase_overview": overview,
        "file_tree": file_tree,
        "labels_map": labels_map,
        "chat_history": [],
    }

    reasoning_llm = get_llm("o3-mini")
    fast_llm = get_llm("gpt-4o")

    agent = OncallAgent(reasoning_llm, fast_llm)
    response = agent.invoke(state)
    print("RCA:", response["rca"])


if __name__ == "__main__":
    main()
