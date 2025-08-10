from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

# Get the API key from environment
api_key = os.getenv("OPENAI_API_KEY")

# Instantiate the LLM
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key
)

# Make a call
response = llm.invoke("What is the capital of Malaysia?")
print(response.content)
