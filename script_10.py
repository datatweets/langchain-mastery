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
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor

# -----------------------------
# 1) Define the math tool
# -----------------------------
@tool
def hypotenuse_length(input: str) -> float:
    """Calculates the hypotenuse of a right-angled triangle.
    Input format: 'a, b' (two side lengths separated by a comma)."""
    # Clean the input - remove quotes and split
    clean_input = input.strip().strip("'\"")
    sides = clean_input.split(',')
    if len(sides) != 2:
        raise ValueError("Please provide exactly two side lengths, e.g. '10, 12'.")

    # Convert to floats after stripping whitespace
    a = float(sides[0].strip())
    b = float(sides[1].strip())

    # a^2 + b^2, then square root
    return math.sqrt(a**2 + b**2)

# -----------------------------
# 2) Model and tools
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0  # deterministic for calculations
)

tools = [hypotenuse_length]

# -----------------------------
# 3) ReAct prompt with required variables
# -----------------------------
template = """
You are a helpful construction assistant. If a calculation is needed, use the available tools.

You have access to the following tools:
{tools}

Use this exact format:
Thought: think about what to do next
Action: the single tool to use, exactly one of [{tool_names}]
Action Input: the input for the action
Observation: the result of the action
...(you can repeat Thought/Action/Observation)...
Thought: I can now answer
Final Answer: the final answer to the user's question

Question: {input}
Thought: {agent_scratchpad}
""".strip()

react_prompt = PromptTemplate.from_template(template)

# -----------------------------
# 4) Create agent + executor
# -----------------------------
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
app = AgentExecutor(agent=agent, tools=tools, verbose=True)

# -----------------------------
# 5) Ask a natural-language query
# -----------------------------
query = "What is the hypotenuse length of a triangle with side lengths of 10 and 12?"

# NOTE: AgentExecutor expects {"input": "..."} by default.
# This will trigger tool use with our hypotenuse_length tool.
result = app.invoke({"input": query})

# Print the final answer
print("\n--- Answer ---")
print(result["output"])
