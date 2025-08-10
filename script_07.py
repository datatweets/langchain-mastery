from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain import hub  # pulls a ready-made ReAct prompt from LangChain Hub
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor

# 1) Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
# 2) LLM (gpt-4o)
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=api_key,
    temperature=0  # deterministic answers for fact lookups
)

# 3) Tools (Wikipedia)
tools = load_tools(["wikipedia"])

# 4) ReAct prompt (standard one from LangChain Hub)
react_prompt = hub.pull("hwchase17/react")

# 5) Build agent + executor
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6) Ask the question (invoke the agent)
question = "How many people live in Tehran?"
result = agent_executor.invoke({"input": question})

# Result keys typically include: 'input', 'output', 'intermediate_steps'
print(result["output"])
