from datetime import datetime, timezone
from typing import Annotated, Dict, List, TypedDict, operator

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import Runnable
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from oncall.agent.models import TaskComplete
from oncall.agent.nodes.utils import get_formatted_labels_map
from oncall.lib.utils import get_llm
from oncall.logs.tool import LogSearchInput, LogSearchTool

LOGQL_EXAMPLE = '{{"service"="<service>"}}'
LOGQL_MULTISERVICE_EXAMPLE = '{{"service"=~"<service1>|<service2>"}}'
EMPTY_BRACES = "{{}}"

PROMPT_PREFIX = """
You are an expert AI assistant that assists engineers debugging production issues. You specifically help by logs relevant to the issue and a description of what types of logs are needed.
"""

PROMPT_TEMPLATE = f"""
Given a request for the type of logs needed for the investigation and all available log labels, your task is to fetch logs relevant to the request. You will do so by outputting your intermediate reasoning for explaining what action you will take and then outputting either a LogSearchInput to read logs from observability API OR a TaskComplete to indicate that you have completed the request.

Guidelines:
- All log queries must be formulated as valid LogQL Loki queries (i.e. selectors wrapped in curly braces {EMPTY_BRACES} followed by keyword searches denoted by "|=" or regex searches denoted by "|~")
- Example LogQL query: {LOGQL_EXAMPLE} |= <keyword> |~ <regex>
- You can inspect multiple service's logs in same query as in this example: {LOGQL_MULTISERVICE_EXAMPLE}
- Prefer relatively simple queries and avoid complex or hyper-specific regexes to avoid empty results
- Avoid querying system or database logs unless there is a good reason to check there.
- Every query must include at least one label/selector in the curly braces, empty selectors (e.g., curly braces with no content such as {EMPTY_BRACES}) are not allowed
- If you query for log level, use keyword or regex search (e.g. |~ "(?i)error" for searching error logs), DO NOT use "level" tag in the query braces only use keywords
- The timezone for start and end dates should be Universal Coordinated Time (UTC).
- If log search is not returning results (as may show in message history), adjust/widen query as needed.
- Aim to end your log search with a broad enough section of logs captured (multiple services around the time of the issue). 
- Do not output `TaskComplete` until you have a broad view of logs captured at least once across multiple services to get full view.
- Be generous on the time ranges and give +/- 15 minutes to ensure you capture the issue (e.g. 4:10am to 4:40am if issue was at 4:25am).
- If there is source code in your previously gathered context, use it to inform your log search.

- DO NOT use XML tags
- If you failed to fetch any context DO NOT give up and output TaskComplete, keep adjusting your query until you get some logs

CURRENT TIME (UTC): {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")}

LOG LABELS (to help inform LogQL queries):
{{labels_map}}

Use the below history of context you already gathered to inform what steps you will take next. DO NOT make the same query twice, it is a waste of the context window.

ISSUE:
{{issue}}

REQUEST:
{{request}}

CONTEXT YOU PREVIOUSLY GATHERED:
{{chat_history}}
"""


class LogSearchOutput(BaseModel):
    reasoning: str = Field(
        ...,
        description="Intermediate reasoning explaining why you will search for the code you search for or are ready to move on to the next step.",
    )
    output: LogSearchInput | TaskComplete


class LogSearch:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            prefix=PROMPT_PREFIX,
            template=PROMPT_TEMPLATE,
            input_variables=[
                "issue",
                "request",
                "labels_map",
                "chat_history",
            ],
        )
        self.chain = self.prompt | self.llm.with_structured_output(
            LogSearchOutput, method="function_calling"
        )

    def invoke(
        self,
        issue: str,
        request: str,
        labels_map: str,
        chat_history: List[str],
    ) -> LogSearchOutput:
        formatted_chat_history = "\n\n".join(
            [f"{i + 1}. {entry}" for i, entry in enumerate(chat_history)]
        )

        response = self.chain.invoke(
            {
                "issue": issue,
                "request": request,
                "labels_map": labels_map,
                "chat_history": formatted_chat_history,
            }
        )
        return response


class LogSearchAgentState(TypedDict):
    first_pass: bool
    issue: str
    request: str
    labels_map: str
    chat_history: Annotated[List[str], operator.add]


class LogSearchAgent(Runnable):
    def __init__(self, llm):
        self.llm = llm
        self.compiled_graph = self.build_graph()

    def log_search(
        self,
        state: LogSearchAgentState,
    ):
        searcher = LogSearch(self.llm)
        response = searcher.invoke(
            issue=state["issue"],
            request=state["request"],
            labels_map=state["labels_map"],
            chat_history=state["chat_history"],
        )

        output = response.output
        if isinstance(output, TaskComplete):
            print("Reasoning: ", response.reasoning)
            print("-- Task Complete --")
        elif isinstance(output, LogSearchInput):
            log_ctx = LogSearchTool()._run(
                start=output.start,
                end=output.end,
                query=output.query,
                limit=output.limit,
            )
            log_ctx = "EMPTY" if not log_ctx else log_ctx

            print("Reasoning: ", response.reasoning)
            print(
                f"Query: {output.query}. Start: {output.start}. End: {output.end}. Limit: {output.limit}"
            )
            print(f"Read logs:\n {log_ctx}")
            print("\n\n")

            chat_message = f"Log search: {output}\n\n Results:\n {log_ctx}"
            return Command(update={"chat_history": [chat_message]}, goto="log_search")

    def build_graph(self) -> CompiledStateGraph:
        node_map = {
            "log_search": self.log_search,
        }

        graph = StateGraph(LogSearchAgentState)
        graph.add_node("log_search", node_map["log_search"])
        graph.add_edge(START, "log_search")

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
        }


def main():
    llm = get_llm("gpt-4o")

    # TODO: LLM figures out labels map for time
    labels_map = get_formatted_labels_map("2025-03-03 03:00:00", "2025-03-03 03:30:00")

    state = {
        "first_pass": True,
        "issue": "User is reporting that they have purchased tickets, payment went through, but ticket later showed as cancelled. This was roughly 2025-03-03 around 3:25am UTC. User's email was something like hello@gmail.com.",
        "request": "Find the logs containing the error being referred to in the issue.",
        "labels_map": labels_map,
        "chat_history": [],
    }

    agent = LogSearchAgent(llm)
    response = agent.invoke(state)
    print(response)


if __name__ == "__main__":
    main()
