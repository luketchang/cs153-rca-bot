from typing import Annotated, Any, Dict, List, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from oncall.agent.models import LogSearchInput, ModuleSelectionInput
from oncall.code.select_modules import ModuleSelector
from oncall.lib.utils import get_llm
from oncall.logs.search import fetch_loki_logs
from oncall.logs.utils import compress_loki_logs


class AgentState(TypedDict):
    """State for the agent."""

    messages: Annotated[Sequence[BaseMessage], "Messages in the conversation so far"]
    next: Annotated[str, "Next node to call"]


@tool
def select_modules_tool(input: ModuleSelectionInput) -> str:
    """
    Select relevant source code files based on a production issue.

    This tool analyzes a codebase and returns the most relevant modules/services
    for investigating a production issue. It helps narrow down which parts of the
    codebase to examine when debugging a problem.
    """
    selector = ModuleSelector(llm=get_llm("gpt-4o"))
    result = selector.select_module_paths(
        input.walkthrough, input.file_tree, input.issue
    )

    # Format the output for readability
    output = "Selected modules for investigation:\n\n"
    for selection in result.selections:
        output += f"- {selection.module}: {selection.reason}\n"

    return output


@tool
def log_search_tool(input: LogSearchInput) -> str:
    """
    Search Grafana Loki logs based on query and time range.

    This tool fetches logs from Grafana Loki based on a query and time range.
    Use it to investigate logs when debugging production issues.
    """
    base_url = "https://logs-prod-021.grafana.net"  # This should be configured properly

    logs = fetch_loki_logs(
        base_url, input.start_time, input.end_time, input.query, input.limit
    )

    if not logs:
        return "No logs found or error fetching logs."

    formatted_logs = compress_loki_logs(logs)
    return "\n".join(formatted_logs)


def create_tools() -> List[BaseTool]:
    """Create the tools for the agent."""
    return [select_modules_tool, log_search_tool]


def create_agent():
    """Create a LangGraph agent with module selection and log search tools."""

    # Create the tools
    tools = create_tools()

    # Create the system prompt
    system_prompt = """
    You are an on-call support agent designed to help engineers debug production issues.
    You have access to two main tools:
    
    1. select_modules_tool: Identifies the most relevant code modules for a given issue
    2. log_search_tool: Searches Grafana Loki logs for relevant information
    
    Given a production issue, your job is to:
    1. Use select_modules_tool to identify the relevant code modules
    2. Use log_search_tool to search for relevant logs
    3. Analyze the results and provide a diagnosis of the issue
    
    Always think step by step and use the tools in a logical sequence.
    """

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create the LLM
    llm = get_llm("o3-mini", prompt)

    # Create the tool executor node
    tool_executor = ToolNode(tools)

    # Define the agent state
    def agent_node(state: AgentState) -> Dict[str, Any]:
        """Process agent state and determine next action."""
        messages = state["messages"]

        # Add agent scratchpad to state
        state["agent_scratchpad"] = []

        # Invoke the LLM
        response = llm.invoke(
            {
                "messages": messages,
                "agent_scratchpad": state.get("agent_scratchpad", []),
            }
        )

        # Parse the response for tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            # Handle tool calls
            tool_call = response.tool_calls[0]
            state["next"] = tool_call.name
            return {
                "messages": messages + [response],
                "next": tool_call.name,
                **tool_call.args,
            }
        else:
            # No tool calls, end the conversation
            return {"messages": messages + [response], "next": END}

    # Create the state graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tool_executor", tool_executor)

    # Add edges
    workflow.add_edge("agent", "tool_executor")
    workflow.add_edge("tool_executor", "agent")

    # Set the entry point
    workflow.set_entry_point("agent")

    # Compile the graph
    return workflow.compile()


def run_agent(
    issue_description: str, walkthrough: str, file_tree: str, log_query: str = None
):
    """
    Run the agent to help diagnose and solve an issue.

    Args:
        issue_description: Description of the production issue
        walkthrough: Description of the codebase/system
        file_tree: File tree structure of the codebase
        log_query: Optional initial log query to run

    Returns:
        The agent's final response
    """
    agent = create_agent()

    # Format the initial message
    initial_message = f"""
    I'm investigating the following production issue:
    
    {issue_description}
    
    Please help me identify the relevant code modules and logs to understand and fix this issue.
    """

    # Create the initial state
    state = {"messages": [HumanMessage(content=initial_message)]}

    # Run the agent
    result = agent.invoke(state)

    # Return the final messages
    return result["messages"]


if __name__ == "__main__":
    # Example usage
    issue = "Users are reporting 401 Unauthorized errors when trying to access the payment API."
    walkthrough = "Our system consists of several microservices: auth, payments, orders, and notifications."
    file_tree = "services/\n  auth/\n  payments/\n  orders/\n  notifications/"

    result = run_agent(issue, walkthrough, file_tree)

    # Print the final response
    for message in result:
        if hasattr(message, "content"):
            print(f"{message.type}: {message.content}")
