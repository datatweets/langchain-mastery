from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")

# Define a chat prompt template with system, human, and AI messages
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a geography expert that returns the colors present in a country's flag."),
        ("human", "France"),
        ("ai", "blue, white, red"),
        ("human", "{country}")
    ]
)

# Create the LLM
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=api_key
)

# Chain the prompt template to the LLM
llm_chain = prompt_template | llm

# Run the chain with a country input
country_name = "Malaysia"
response = llm_chain.invoke({"country": country_name})

# Print the result
print(response.content)
