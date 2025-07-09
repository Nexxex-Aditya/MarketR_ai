# This will help to chains your agents in the correct order: Forecast > News > Simulate > Summarize
# here TypeDict is used to define what information each step passes forward
# When build_synergistic_workflow() is called, it returns the compiled LangGraph workflow ready to run asynchronously.

from typing import Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

# Import agents
from agents.forecast_agent import forecast_agent_node
from agents.simulation_agent import simulation_agent_node
from agents.summary_agent import summary_agent_node
from agents.news_agent import news_search_node  # wraps your LangGraph-compatible news tool

# Define shared state structure
class AgentState(TypedDict):
    query: str
    forecast_result: Dict[str, Any]
    forecast_summary: str
    news_results: list
    news_summary: str
    simulation_result: list
    simulation_summary: str
    summary_text: str
    ppt_file: str
    news_query: str  # optional: refined from query

# Build the LangGraph workflow
def build_synergistic_workflow():
    graph = StateGraph(AgentState)

    # Add all the agent nodes
    graph.add_node("forecast", forecast_agent_node)
    graph.add_node("news", news_search_node)
    graph.add_node("simulate", simulation_agent_node)
    graph.add_node("summarize", summary_agent_node)

    # Define the execution flow
    graph.set_entry_point("forecast")
    graph.add_edge("forecast", "news")
    graph.add_edge("news", "simulate")
    graph.add_edge("simulate", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()
