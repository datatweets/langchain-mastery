from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

# Get the API key from environment
api_key = os.getenv("OPENAI_API_KEY")

# Create a prompt template
template = "You are an artificial intelligence assistant. Answer the question clearly and concisely.\nQuestion: {question}"
prompt = PromptTemplate.from_template(template)

# Instantiate the LLM
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=api_key   # use api_key here for langchain_openai
)

# Create a chain by piping prompt → llm
llm_chain = prompt | llm

# Question to send
question = "What is the capital of Malaysia?"

# Invoke the chain and print result
response = llm_chain.invoke({"question": question})
print(response.content)
