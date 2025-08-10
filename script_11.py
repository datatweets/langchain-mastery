# tool_math_agent.py
import os
import math
from dotenv import load_dotenv

# Disable LangSmith tracing (optional) BEFORE importing langchain bits
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"
for var in ["LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "LANGCHAIN_ENDPOINT", "LANGSMITH_ENDPOINT", "LANGCHAIN_PROJECT"]:
    os.environ.pop(var, None)

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# -----------------------------
# 1) Define the math tool
# -----------------------------
@tool
def hypotenuse_length(input: str) -> float:
    """Calculates the hypotenuse of a right-angled triangle.
    Input format: 'a, b' (two side lengths separated by a comma)."""
    clean_input = input.strip().strip("'\"")
    sides = clean_input.split(',')
    if len(sides) != 2:
        raise ValueError("Please provide exactly two side lengths, e.g. '10, 12'.")
    a = float(sides[0].strip())
    b = float(sides[1].strip())
    return math.sqrt(a**2 + b**2)

# -----------------------------
# 2) Model and tools
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)

tools = [hypotenuse_length]

# -----------------------------
# 3) ReAct CHAT prompt with history
# -----------------------------
react_chat_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful construction assistant. Use the available tools when a calculation is needed.

You have access to the following tools:
{tools}

Follow this EXACT format in your replies:
Thought: think about what to do next
Action: the single tool to use, exactly one of [{tool_names}]
Action Input: the input for the action
Observation: the result of the action
...(you can repeat Thought/Action/Observation)...
Thought: I can now answer
Final Answer: the final answer to the user's question

IMPORTANT:
- Never skip 'Final Answer:' when you give the final result.
- Only call tools using the Action/Action Input lines in the format above."""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Question: {input}"),
    ("assistant", "{agent_scratchpad}"),
])

# -----------------------------
# 4) Create agent + executor (add parsing recovery)
# -----------------------------
agent = create_react_agent(llm=llm, tools=tools, prompt=react_chat_prompt)

# ✅ Key fix: let the agent recover from output parsing issues
app = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,           # <--- add this
    max_iterations=6,                     # optional safety valves
    early_stopping_method="generate",     # optional: generate a final answer when hitting limits
)

# -----------------------------
# 5) Run with REAL conversation history
# -----------------------------
chat_history = []

# (a) First query
first_query = "What is the hypotenuse length of a triangle with side lengths of 10 and 12?"
first_result = app.invoke({"input": first_query, "chat_history": chat_history})
print("\n--- Answer (first query) ---")
print(first_result["output"])

chat_history.append(HumanMessage(content=first_query))
chat_history.append(AIMessage(content=first_result["output"]))

# (b) Comparison print
query = "What is the value of the hypotenuse for a triangle with sides 3 and 5?"
response = app.invoke({"input": query, "chat_history": chat_history})
print("\n--- Comparison print (input vs output) ---")
print({"user_input": query, "agent_output": response["output"]})

chat_history.append(HumanMessage(content=query))
chat_history.append(AIMessage(content=response["output"]))

# (c) Follow-up using full history
new_query = "What about one with sides 12 and 14?"
follow_up = app.invoke({"input": new_query, "chat_history": chat_history})

# For teaching: show Human/AI-only log
_display_history = chat_history + [HumanMessage(content=new_query), AIMessage(content=follow_up["output"])]
filtered_messages = [
    f"{msg.__class__.__name__}: {msg.content}"
    for msg in _display_history
    if isinstance(msg, (HumanMessage, AIMessage)) and msg.content.strip()
]
print("\n--- Follow-up using history ---")
print({"user_input": new_query, "agent_output": filtered_messages})

# Update history if continuing
chat_history.append(HumanMessage(content=new_query))
chat_history.append(AIMessage(content=follow_up["output"]))
