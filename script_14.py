# multi_tool_graph_chatbot.py
# Full lesson demo: multiple tools + LangGraph + memory + PNG diagram export

import os
from dotenv import load_dotenv

load_dotenv()

# ---- LLM ----
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2,
)

# ---- Tools ----
from langchain.tools import tool
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

# Wikipedia tool (top-1 result)
api_wrapper = WikipediaAPIWrapper(top_k_results=1)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Tool that invokes the LLM inside the tool body
@tool
def historical_events(date_input: str) -> str:
    """Provide a list of important historical events for a given date in any format."""
    try:
        # Ask the LLM to interpret the date and enumerate events
        response = llm.invoke(
            f"List important historical events that occurred on {date_input}. "
            f"Answer concisely with bullets and brief dates."
        )
        return response.content
    except Exception as e:
        return f"Error retrieving events: {str(e)}"

# Palindrome checker tool
@tool
def palindrome_checker(text: str) -> str:
    """Check if a word or phrase is a palindrome."""
    cleaned_text = "".join(ch.lower() for ch in text if ch.isalnum())
    if cleaned_text == cleaned_text[::-1]:
        return f"The phrase or word '{text}' is a palindrome."
    else:
        return f"The phrase or word '{text}' is not a palindrome."

# Bind all tools to the LLM (so it can request tool calls)
tools = [wikipedia_tool, palindrome_checker, historical_events]
model_with_tools = llm.bind_tools(tools)

# ---- LangGraph wiring ----
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Tool node (executes tool calls requested by the model)
tool_node = ToolNode(tools=tools)

# Decide whether to continue (i.e., if last message contains tool calls)
def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END

# Call the model; if last AI message already contains a tool response, return it
from langchain_core.messages import AIMessage, HumanMessage

def call_model(state: MessagesState):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        # Return only the tool's response if present
        return {"messages": [AIMessage(content=last_message.tool_calls[0]["response"])]}
    # Otherwise, call the tool-enabled model with the full history
    return {"messages": [model_with_tools.invoke(state["messages"])]}

# Build the workflow graph
workflow = StateGraph(MessagesState)

# Nodes
workflow.add_node("chatbot", call_model)
workflow.add_node("tools", tool_node)

# Edges
workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges("chatbot", should_continue, ["tools", END])
workflow.add_edge("tools", "chatbot")

# Memory + compile
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ---- Save diagram(s) to PNG ----
def save_graph_png(graph_app, out_path: str):
    try:
        png_bytes = graph_app.get_graph().draw_mermaid_png()
        with open(out_path, "wb") as f:
            f.write(png_bytes)
        print(f"Diagram saved to {out_path}")
        # If running in a notebook, also show inline (best-effort)
        try:
            from IPython.display import Image, display
            display(Image(png_bytes))
        except Exception:
            pass
    except Exception:
        print("Diagram generation requires additional dependencies.")

save_graph_png(app, "multi_tool_graph_with_memory.png")

# ---- Streaming helpers ----
config = {"configurable": {"thread_id": "1"}}

def multi_tool_output(query: str):
    """Single-turn: stream the agent's answer for a single query."""
    inputs = {"messages": [HumanMessage(content=query)]}
    print(f"User: {query}")
    print("Agent: ", end="", flush=True)
    for msg, metadata in app.stream(inputs, config, stream_mode="messages"):
        if msg.content and not isinstance(msg, HumanMessage):
            print(msg.content, end="", flush=True)
    print("\n")

def user_agent_multiturn(queries):
    """Multi-turn: prints user query then streams only agent messages per turn."""
    for query in queries:
        print(f"User: {query}")
        agent_chunks = []
        for msg, metadata in app.stream(
            {"messages": [HumanMessage(content=query)]},
            config,
            stream_mode="messages",
        ):
            if msg.content and not isinstance(msg, HumanMessage):
                agent_chunks.append(msg.content)
        print("Agent: " + "".join(agent_chunks) + "\n")

# ---- Demo ----
if __name__ == "__main__":
    # Try the different tools
    multi_tool_output("Is `may a moody baby doom a yam` a palindrome?")
    multi_tool_output("What happened on 20th July, 1969?")
    multi_tool_output("Summarize the Eiffel Tower in 2 sentences.")

    # Multi-turn conversation with memory
    queries = [
        "Is `stressed desserts?` a palindrome?",
        "What about the word `kayak`?",
        "What happened on the May 8th, 1945?",
        "What about 9 November 1989?",
    ]
    user_agent_multiturn(queries)
