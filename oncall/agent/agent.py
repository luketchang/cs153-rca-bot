import os
from datetime import datetime, timezone
from pathlib import Path

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from oncall.code.common import collect_files, generate_tree_string
from oncall.code.tool import RipgrepMultiSearchTool
from oncall.lib.utils import get_llm
from oncall.logs.labels import build_labels_map
from oncall.logs.tool import LogSearchTool

# Constants
DEFAULT_BASE_URL = "https://logs-prod-021.grafana.net"

# Cache for storing context data to avoid refetching
_context_cache = {"file_tree": None, "codebase_overview": None, "labels_map": None}


def load_file_tree(repo_path):
    """
    Load the file tree string for a given repository path.
    Uses the same functionality as the code module to generate the file tree.

    Args:
        repo_path: Path to the repository root

    Returns:
        A string representation of the file tree
    """
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


def load_labels_map(base_url=DEFAULT_BASE_URL, hours_back=12):
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


def create_agent(repo_path, codebase_overview_path, input_text):
    # Load context only once
    file_tree = load_file_tree(repo_path)
    codebase_overview = load_codebase_overview(codebase_overview_path)
    labels_map = load_labels_map()

    system_message = """
    You are a proactive oncall debugging agent tasked with assisting engineers in diagnosing and resolving production issues by identifying the precise root cause.

    You are a proactive oncall debugging agent tasked with assisting engineers in diagnosing and resolving production issues by identifying the precise root cause. You have the capability to search both code and logs to gather comprehensive information on system behavior and potential problems.

    General Guidelines:
    - Every investigation must start with the following steps:
        1) Broad code search across at least 3 different top level services likely relevant to the issue
        2) A narrow log search to identify the relevant time frame and services involved
    - Always corroborate your findings by cross-referencing both code and logs; avoid drawing conclusions from a single source.
    - Continue gathering context until you are confident in pinpointing a specific root cause—whether it be a system resource issue or a bug in the application code.
    - A root cause cannot be, for example, "Synchronization Issues" or "Performance Issues" it must be way more specific than that with concrete explanation.
    - Do not conclude the investigation/finish agent chain until you have a fully formulated, actionable root cause and no further investigative steps remain
    - Finish by proposing a concrete resolution in the form of code changes or system configuration adjustments. You are not done until you can do this.

    Code Search:
    - You have access to the entire codebase file tree provided below.
    - Use the ripgrep tool to search for code snippets. Inputs should include a list of directory paths and a corresponding regex for each directory.
    - Every search must specify at least a directory and a regex (empty searches are not permitted). Default to using "." regex early on if you are just exploring codebase functionality.
    - Refer to the provided codebase overview for guidance on which areas to inspect.
    - Utilize the codebase path to construct absolute paths for the ripgrep tool mappings—ABSOLUTE PATHS MUST BE USED.
    ABSOLUTE PATHS MUST BE USED
    - Make wide searches to capture context from at least three top-level service directories. Avoid hyper-specific regexes in the early stages.
    - Use regex to filter out files like package-lock.json, __pycache__, etc and other .gitignored files
    - Prioritize context from different services, never query for same service multiple times in same ripgrep tool call.
    - Never target individual files with overly specific regexes; focus on directory-wide searches.
    - Prevent redundant searches by avoiding duplicate retrieval of the same code context.

    Log Search:
    - All log queries must be formulated as valid LogQL queries (i.e. selectors in curly braces followed by keyword searches denoted by "|=")
    - Use the loki_search tool for querying logs
    - Construct queries using the provided labels to ensure effective log retrieval.
    - You are only allowed to use basic labels (e.g. service or app names) and simple keywords to in the query, regexes are NOT allowed in log queries
    - Every query must include at least one label/selector; empty selectors (e.g., curly braces with no content) are not allowed.
    - Refrain from using log levels in queries as they are currently non-functional.
    - Initially narrow the search to specific services to isolate the problematic time frame or service, then expand to include all relevant services to observe key inter-service interactions.
    - The expected timezone for start and end dates is Universal Coordinated Time (UTC).
    """

    # Format the system message with context
    task_message = f"""
    CODEBASE PATH: {repo_path}
    
    CODEBASE OVERVIEW:
    {codebase_overview}

    FILE TREE:
    {file_tree}
    
    LOG LABELS:
    {format_labels_map(labels_map)}

    CURRENT TIME (UTC): {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")}

    Now start investigating the below issue and find the root cause.

    ISSUE:
    {input_text}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("placeholder", "{chat_history}"),
            ("human", task_message),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    model = get_llm("gpt-4o")
    tools = [LogSearchTool(), RipgrepMultiSearchTool()]

    agent = create_tool_calling_agent(model, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def create_and_run_agent(
    input_query, repo_path, codebase_overview_path, chat_history=None
):
    if chat_history is None:
        chat_history = []

    agent_executor = create_agent(repo_path, codebase_overview_path, input_query)
    return agent_executor.invoke({"input": input_query, "chat_history": chat_history})


if __name__ == "__main__":
    # Example usage
    repo_path = Path("/Users/luketchang/code/Microservices_Udemy/ticketing")

    codebase_overview_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "codebase_overviews",
        "typescript-microservices.txt",
    )

    # issue = """
    # User is reporting that they have purchased tickets, payment went through, but ticket later showed as cancelled. This was roughly 2-3 days ago. User's email was something like hello@gmail.com.
    # """

    issue = """
    Getting some "order not found" errors in the payments service which is causing crashes. One of these happened within the last few hours. Why is this happening? The user was luke@gmail.com or luke@email.com.
    """

    # issue = """
    # A user just reported an issue one hour or two ago. They said they have the ticket but there's an order on their screen that also says cancelled. So they're not sure if they have the ticket or not. What happened, is there a bug?
    # """

    # Example with chat history
    result = create_and_run_agent(
        issue,
        repo_path,
        codebase_overview_path,
        chat_history=[],
    )
    print(result)
