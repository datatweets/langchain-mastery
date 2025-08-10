from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from dotenv import load_dotenv
import os

# Load env vars
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# --- Few-shot examples about a GitHub user (Lina) ---
examples = [
    {"question": "How many public repositories does Lina have on GitHub?", "answer": "42"},
    {"question": "How many total stars has Lina received across repositories?", "answer": "1,580"},
    {"question": "Which language does Lina use most on GitHub?", "answer": "Python"},
]

# How each example is rendered
example_prompt = PromptTemplate.from_template(
    "Question: {question}\nAnswer: {answer}"
)

# Build the few-shot prompt
prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}\nAnswer:",
    input_variables=["input"],
)

# LLM (works with gpt-4o)
llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

# Chain prompt -> llm
chain = prompt_template | llm

# Ask a new question (model will mimic the examples' style)
user_question = "What is Lina's primary programming language on GitHub?"
response = chain.invoke({"input": user_question})



# (Optional) inspect the final prompt that was sent
formatted = prompt_template.invoke({"input": user_question})
print("\n--- Formatted Prompt ---\n")
print(formatted.to_string(),response.content)  # Expected pattern-consistent answer, e.g., "Python"