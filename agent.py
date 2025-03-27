import operator
from typing import TypedDict, Dict, Any, List
try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated
import logging
import uuid
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    query: str
    proposed_response: str
    human_input: Annotated[List[str], operator.add]  # Ensure it's a list that accumulates
    final_response: str
    messages: Annotated[List[Any], operator.add]

def start_node(state: AgentState) -> Dict[str, Any]:
    logger.info(f"Starting query: {state['query']}")
    return {
        "messages": [("user", state["query"])],
        "human_input": [],  # Initialize human_input to ensure it exists
    }

def agent_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    proposed_response = f"Agent's initial response to '{query}': This is a draft response. Please review and provide feedback."
    logger.info(f"Proposed response: {proposed_response}")
    return {
        "proposed_response": proposed_response,
        "messages": [
            ("ai", proposed_response),
            ("system", "HUMAN INTERVENTION REQUIRED: Please review the proposed response and provide your feedback or modifications.")
        ]
    }

def finalize_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Finalizing response")
    human_feedback = state.get("human_input", [])
    
    if human_feedback:
        final_response = human_feedback[-1]  # Use the last human-provided response
        logger.info(f"Using human-refined response: {final_response}")
    else:
        final_response = state.get("proposed_response", "No response generated")
        logger.info(f"Using original agent response: {final_response}")

    return {
        "final_response": final_response,
        "messages": [("system", f"Final Response: {final_response}")]
    }

workflow = StateGraph(AgentState)
workflow.add_node("start", start_node)
workflow.add_node("agent", agent_node)
workflow.add_node("finalize", finalize_node)

workflow.set_entry_point("start")
workflow.add_edge("start", "agent")
workflow.add_edge("agent", "finalize")
workflow.add_edge("finalize", END)

memory = MemorySaver()
graph_app = workflow.compile(
    interrupt_before=["finalize"],
    checkpointer=memory
)

def run_agent(query: str, human_input: str = "", thread_id: str = None) -> Dict[str, Any]:
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    is_initial_run = not human_input.strip()
    initial_input = {"query": query, "human_input": []} if is_initial_run else None

    logger.info("--- Running initial stream ---")
    events = list(graph_app.stream(initial_input, config=config, stream_mode="values"))

    logger.info("Agent response generated. Waiting for human review.")

    if human_input.strip():
        logger.info(f"Updating state with human input: '{human_input}'")
        graph_app.update_state(config, {"human_input": [human_input]})  # Append to human_input
        graph_app.update_state(config, {"messages": [("human", human_input)]})

    logger.info("--- Resuming execution ---")
    final_state_events = list(graph_app.stream(None, config=config, stream_mode="values"))

    return final_state_events[-1] if final_state_events else {}
