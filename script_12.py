
import os
from dotenv import load_dotenv

# --- Load environment (expects OPENAI_API_KEY in .env) ---
load_dotenv()

# --- LangChain / LangGraph imports ---
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

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
    # An accumulating list of chat messages; LangGraph will merge/append for us
    messages: Annotated[list, add_messages]

# -----------------------------
# 3) Build the graph
# -----------------------------
graph_builder = StateGraph(State)

def chatbot(state: State):
    """
    Single node: call the LLM with running 'messages' and return the AI's reply.
    """
    ai_message = llm.invoke(state["messages"])
    return {"messages": [ai_message]}

# Add node and edges
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile()

# -----------------------------
# 4) Multi-turn conversation with state
# -----------------------------
# Seed the conversation with a concise, high-school-friendly style
system_style = ("system", "Answer concisely (2–4 sentences) for high-school students.")

state: State = {"messages": [system_style]}  # running memory

def ask(user_input: str):
    """
    Stream a turn through the graph, print the latest agent message,
    and update the running state with full message history.
    """
    print(f"\nUser: {user_input}")
    # Merge prior state + new user message; `add_messages` will append properly
    for event in graph.stream({**state, "messages": [("user", user_input)]}):
        for item in event.values():
            msgs = item.get("messages", [])
            if msgs:
                last = msgs[-1]
                # Handle either LangChain Message objects or raw tuples
                content = getattr(last, "content", last[1] if isinstance(last, (tuple, list)) else str(last))
                print("Agent:", content)
                # Update state so next turn has full history
                state["messages"] = msgs

# -----------------------------
# 5) Demo runs
# -----------------------------
if __name__ == "__main__":
    ask("Who is Ada Lovelace?")
    ask("Name her collaborator and what machine it was.")  # demonstrates memory

    # -------------------------
    # 6) Generate and save the graph diagram
    # -------------------------
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        out_path = "chatbot_graph.png"
        with open(out_path, "wb") as f:
            f.write(png_bytes)
        print(f"\nDiagram saved to {out_path}")
        if _HAVE_IPY:
            display(Image(png_bytes))  # inline display if in a notebook
    except Exception:
        print("Diagram generation requires additional dependencies.")
