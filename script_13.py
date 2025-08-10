# wikipedia_chatbot_graph.py
# Chatbot with Wikipedia tool via LangGraph, conversation memory, and PNG diagram export.

import os
from dotenv import load_dotenv

# --- Load env (expects OPENAI_API_KEY in .env) ---
load_dotenv()

# --- Core LangChain / LangGraph imports ---
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Wikipedia tool
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

# Tool routing in LangGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Memory checkpointer
from langgraph.checkpoint.memory import MemorySaver

# (Optional) inline display if running in a notebook; safe to import lazily later
try:
    from IPython.display import Image, display  # noqa: F401
    _HAVE_IPY = True
except Exception:
    _HAVE_IPY = False


# -----------------------------
# 1) Define the LLM
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2,
)


# -----------------------------
# 2) Define the State schema
# -----------------------------
class State(TypedDict):
    # Accumulating list of messages; LangGraph merges/appends via add_messages
    messages: Annotated[list, add_messages]


# -----------------------------
# 3) Build Wikipedia tool + bind to model
# -----------------------------
# Initialize Wikipedia API wrapper to fetch only the top result
api_wrapper = WikipediaAPIWrapper(top_k_results=1)

# Create the Wikipedia query tool and make a tools list
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
tools = [wikipedia_tool]

# Bind tools to the LLM so it can emit tool-calls in responses
llm_with_tools = llm.bind_tools(tools)


# -----------------------------
# 4) Graph: nodes and edges
# -----------------------------
graph_builder = StateGraph(State)

# Chatbot node that lets the model decide whether to call the tool
def chatbot(state: State):
    # Pass the current messages to the tool-enabled model
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Add chatbot node
graph_builder.add_node("chatbot", chatbot)

# Create a ToolNode to execute tool calls when the model requests them
tool_node = ToolNode(tools=[wikipedia_tool])
graph_builder.add_node("tools", tool_node)

# Conditional routing:
# - If the model called a tool, route to the "tools" node
# - Otherwise route toward END
graph_builder.add_conditional_edges("chatbot", tools_condition)

# Connect tools back to chatbot (to continue after tool results)
graph_builder.add_edge("tools", "chatbot")

# Start and end of the flow
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph (initial, without memory)
graph = graph_builder.compile()


# -----------------------------
# 5) Save & (optionally) display the graph diagram
# -----------------------------
def save_graph_png(g, out_path: str = "wikipedia_chatbot_graph.png"):
    try:
        png_bytes = g.get_graph().draw_mermaid_png()
        with open(out_path, "wb") as f:
            f.write(png_bytes)
        print(f"Diagram saved to {out_path}")
        if _HAVE_IPY:
            display(Image(png_bytes))
    except Exception:
        print("Diagram generation requires additional dependencies.")

# Save a diagram for the initial graph (without memory)
save_graph_png(graph, "wikipedia_chatbot_graph_nomemory.png")


# -----------------------------
# 6) Add memory and recompile
# -----------------------------
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Save a diagram for the memory-enabled graph
save_graph_png(graph, "wikipedia_chatbot_graph_with_memory.png")


# -----------------------------
# 7) Stream responses with memory
# -----------------------------
def stream_memory_responses(user_input: str):
    """
    Streams events from the memory-enabled graph. Uses a single thread_id so
    follow-up questions can rely on previous turns automatically.
    """
    config = {"configurable": {"thread_id": "single_session_memory"}}

    # Stream events; LangGraph yields partial states as the graph runs
    for event in graph.stream({"messages": [("user", user_input)]}, config):
        for value in event.values():
            if "messages" in value and value["messages"]:
                # Print the content of the latest message only (cleaner for terminal)
                last = value["messages"][-1]
                content = getattr(last, "content", last[1] if isinstance(last, (tuple, list)) else str(last))
                print("Agent:", content)


# -----------------------------
# 8) Demo
# -----------------------------
if __name__ == "__main__":
    # First question can be answered from model or by calling Wikipedia
    stream_memory_responses("Tell me about the Eiffel Tower.")
    # Follow-up relies on memory (no need to restate context)
    stream_memory_responses("Who built it?")
