import os
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.prompts import PromptTemplate

from oncall.agent.models import AgentAction, AttemptComplete, IntermediateReasoning
from oncall.code.common import collect_files, generate_tree_string
from oncall.code.tool import CodeSearchInput, CodeSearchTool
from oncall.constants import GRAFANA_URL
from oncall.lib.utils import get_llm
from oncall.logs.labels import build_labels_map
from oncall.logs.tool import LogSearchInput, LogSearchTool

PROMPT_PREFIX = """
You are a proactive oncall debugging agent tasked with assisting engineers in diagnosing and resolving production issues by identifying the precise root cause.

Given a codebase/system walkthrough, the codebase file tree, and an issue report (either from system alert or customer support ticket), a set of log search labels, and the current time, your task is to gather enough context to figure out the root cause of the issue and provide an analysis and solution. You will be able to gather code and log context by outputting CodeSearchInput and LogSearchInput actions. You will also be able to output IntermediateReasoning blocks to synthesize information or hypotheses on the context you already have. Only once you have gathered enough of both types of context, are confident in your reasoning, and previously instructed yourself in your last IntermediateReasoning step that you are ready for final answer should you output an AttemptComplete action.

General Guidelines:
- ALWAYS START WITH CODE SEARCH, DO NOT DO LOGS BEFORE CODE
- When working with microservices, be sure to expand your code searches over time to include the other services the interact with the original service of interest that may have been experiencing issues. Inter-service interactions are key to understanding the full context of an issue.
- When searching logs, also make sure you gather logs across multiple services to get full picture across services
- Rely on your intermediate reasoning when choosing what action to perform next
- Continue gathering context until you are confident in pinpointing a specific root cause—whether it be a system resource issue or a bug in the application code.
- A root cause cannot be, for example, "Synchronization Issues" or "Performance Issues" it must be way more specific than that with concrete explanation.
- Do not conclude the investigation/finish agent chain until you have a fully formulated, actionable root cause and no further investigative steps remain
- Finish by proposing a concrete resolution in the form of code changes or system configuration adjustments. You are not done until you can do this.
- If your tool calls to gather context are not returning any results, loosen the search criteria until you find some results (this can include keywords, time ranges, etc)
- If some searches are failing, try expanding their scope by removing keywords or widening time ranges. But gradually, do not make the search so broad that it overflows the context window (e.g. searching for all files recursively from root).
- AVOID READING THE SAME CONTEXT MULTIPLE TIMES, IT WASTES THE CONTEXT WINDOW.


Code (CodeSearchInput):
- You have access to the entire codebase file tree provided below.
- Use the ripgrep tool to search for code snippets. Inputs should include a list of directory paths you want to read content from.
- Refer to the provided codebase overview for guidance on which areas to inspect.
- Utilize the codebase path to construct absolute paths for the ripgrep tool mappings—ABSOLUTE PATHS MUST BE USED.
- Prioritize context from different services, never query for same service multiple times in same ripgrep tool call.
- Prevent redundant searches by avoiding duplicate retrieval of the same code context.

Logs (LogSearchInput):
- All log queries must be formulated as valid LogQL Loki queries (i.e. selectors wrapped in curly braces {{}} followed by keyword searches denoted by "|=")
- Example LogQL query: {{"service"="<service>"}} |="<keyword>"
- Construct queries using the provided labels to ensure effective log retrieval.
- You are only allowed to use basic labels (e.g. service or app names) and simple keywords to in the query, regexes are NOT allowed in log queries
- Every query must include at least one label/selector; empty selectors (e.g., curly braces with no content) are not allowed.
- Refrain from using log levels in queries as they are currently non-functional.
- Initially narrow the search to specific services to isolate the problematic time frame or service, then expand to include all relevant services to observe key inter-service interactions.
- The expected timezone for start and end dates is Universal Coordinated Time (UTC).
- If log search is not returning results (as may show in message history), adjust query as needed.
- Be generous on the time ranges and give +/- 15 minutes to ensure you capture the issue (e.g. 4:10am to 4:40am if issue was at 4:25am).

Intermediate Reasoning (IntermediateReasoning):
- Use this action to provide a summary of the logs and code you have gathered so far, what issues you may be encountering if any, and what you plan to do next to further investigate the issue.
- Do not use multiple intermediate reasoning steps in a row, use it as whiteboard after taking one or more actions.
- Ask yourself if you have gathered enough code or log context from different types of services or components that may be involved in the root cause, NOT just the ones immediately affected. Especially in microservices, the root cause may not be in the service that is failing, but in another service that is interacting with it. So make sure to explore other services as part of your next steps before completing the root cause analysis.
- Only output next step is to provide root cause analysis unless you have fully finished exploring and have enough context to do so.

Attempt Completion (AttemptComplete):
- Only output this response if all of the following are true:
    - You have gathered context from BOTH logs and code (not just one or the other)
    - You have thoroughly explored the logs and code from multiple angles and services (not just the ones immediately affected)
    - There was a previous intermediate reasoning step that indicated you were ready to give a final answer
    - You are fully confident you have a complete and accurate root cause analysis
- If you are unsure, do not return this output and instead use another tool to gather more context OR write down your intermediate reasoning.
- If you are struggling with tool use, do not pick this option and give up, keep trying to use the tools until you have gathered enough context for a confident root cause analysis.
- DO NOT choose this option unless you have gathered significant context from BOTH logs and code already. If that is not the case, output a code search or log search action to get more.
- DO NOT choose this option if the last message in existing context is a directive to go investigate logs or code, as that indicates you are not ready to give a final answer and have more context to gather.
"""

PROMPT_TEMPLATE = f"""
CURRENT TIME (UTC): {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")}

CODEBASE PATH: {{repo_path}}

CODEBASE OVERVIEW:
{{codebase_overview}}

FILE TREE:
{{file_tree}}

LOG LABELS:
{{labels_map}}

Use the below history of context you already gathered to inform what steps you will take next. DO NOT make the same query twice, it is a waste of the context window.

CONTEXT YOU PREVIOUSLY GATHERED:
{{chat_history}}

ISSUE:
{{input_text}}
"""


_context_cache = {"file_tree": None, "codebase_overview": None, "labels_map": None}


def load_file_tree(repo_path):
    if _context_cache["file_tree"] is not None:
        return _context_cache["file_tree"]

    try:
        # Define allowed file extensions for code files
        allowed_extensions = (
            ".py",
            ".js",
            ".ts",
            ".java",
            ".go",
            ".rs",
            ".yaml",
            ".cs",
        )

        # Collect the file tree (list of file paths) from the repo
        repo_file_tree, _ = collect_files(
            repo_path, allowed_extensions, repo_root=repo_path
        )

        # Generate the tree string representation
        file_tree_str = generate_tree_string(repo_file_tree)

        # Cache the result
        _context_cache["file_tree"] = file_tree_str

        return file_tree_str
    except Exception as e:
        print(f"Error loading file tree: {e}")
        return "File tree could not be loaded."


def load_codebase_overview(overview_path):
    if _context_cache["codebase_overview"] is not None:
        return _context_cache["codebase_overview"]

    try:
        with open(overview_path, "r") as f:
            overview = f.read()
        _context_cache["codebase_overview"] = overview
        return overview
    except Exception as e:
        print(f"Error loading codebase overview: {e}")
        return "Codebase overview could not be loaded."


def load_labels_map(base_url=GRAFANA_URL, hours_back=12):
    if _context_cache["labels_map"] is not None:
        return _context_cache["labels_map"]

    try:
        labels_map = build_labels_map(base_url, hours_back)
        _context_cache["labels_map"] = labels_map
        return labels_map
    except Exception as e:
        print(f"Error building labels map: {e}")
        return {}


def format_labels_map(labels_map):
    result = "Available Loki log labels and values:\n"
    for label, values in labels_map.items():
        result += f"  {label}: {', '.join(values[:10])}"
        if len(values) > 10:
            result += f" (and {len(values) - 10} more)"
        result += "\n"
    return result


def print_divider(i):
    print(f"\n{'=' * 20} Step {i} {'=' * 20}\n")


class OncallAgent:
    def __init__(self, llm=get_llm("claude-3-5-sonnet-20241022")):
        self.llm = llm
        self.prompt = PromptTemplate(
            prefix=PROMPT_PREFIX,
            template=PROMPT_TEMPLATE,
            input_variables=[
                "repo_path",
                "codebase_overview",
                "file_tree",
                "labels_map",
                "input_text",
                "chat_history",
            ],
        )
        self.context_cache = {}

    def run(self, input_query, repo_path, codebase_overview_path, chat_history=None):
        if chat_history is None:
            chat_history = []

        self.context_cache["file_tree"] = load_file_tree(repo_path)
        self.context_cache["codebase_overview"] = load_codebase_overview(
            codebase_overview_path
        )
        self.context_cache["labels_map"] = format_labels_map(load_labels_map())

        chain = self.prompt | self.llm.with_structured_output(AgentAction)

        counter = 1
        while True:
            print_divider(counter)

            formatted_chat_history = "\n\n".join(
                [f"{i + 1}. {entry}" for i, entry in enumerate(chat_history)]
            )
            response: AgentAction = chain.invoke(
                {
                    "repo_path": repo_path,
                    "codebase_overview": self.context_cache["codebase_overview"],
                    "file_tree": self.context_cache["file_tree"],
                    "labels_map": self.context_cache["labels_map"],
                    "input_text": input_query,
                    "chat_history": formatted_chat_history,
                }
            )

            action = response.action
            if isinstance(action, AttemptComplete):
                print(f"Root cause analysis complete. Analysis:\n {action.root_cause}")
                chat_history.append(action.root_cause)
                return response
            elif isinstance(action, CodeSearchInput):
                tool = CodeSearchTool()
                code_ctx = tool._run(action.directories)
                code_ctx = "EMPTY" if not code_ctx else code_ctx
                result = f"Code search: {action.directories}\n\n Results:\n {code_ctx}"
                print(result)
                chat_history.append(result)
            elif isinstance(action, LogSearchInput):
                tool = LogSearchTool()
                log_ctx = tool._run(
                    action.query,
                    action.start,
                    action.end,
                    action.limit,
                )
                log_ctx = "EMPTY" if not log_ctx else log_ctx
                result = f"Log search: {action.query} {action.start} {action.end}\n\n Results:\n {log_ctx}"
                print(result)
                chat_history.append(result)
            elif isinstance(action, IntermediateReasoning):
                result = f"Intermediate reasoning: {action.reasoning}"
                print(result)
                chat_history.append(result)

            counter += 1


def main():
    repo_path = Path("/Users/luketchang/code/Microservices_Udemy/ticketing")
    overview_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "codebase_overviews",
        "typescript-microservices.txt",
    )

    # issue = """
    # Getting some "order not found" errors in the payments service which is causing crashes. One of these happened around 03-06-2025 around 4am UTC. Why is this happening?
    # """

    issue = """
    User is reporting that they have purchased tickets, payment went through, but ticket later showed as cancelled. This was roughly 2025-03-03 around 3:25am UTC. User's email was something like hello@gmail.com.
    """

    # Initialize and run the agent
    agent = OncallAgent()
    result = agent.run(
        input_query=issue,
        repo_path=repo_path,
        codebase_overview_path=overview_path,
    )

    return result


if __name__ == "__main__":
    main()
