from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# LLM (gpt-4o)
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=api_key,
    temperature=0.3,
)

# 1) Prompt to draft a step-by-step learning plan for the activity
learning_prompt = PromptTemplate(
    input_variables=["activity"],
    template="I want to learn how to {activity}. "
             "Suggest a clear, step-by-step learning plan with milestones and practice tasks."
)

# 2) Prompt to compress that plan into a concise one-week schedule
time_prompt = PromptTemplate(
    input_variables=["learning_plan"],
    template=(
        "I only have one week. Create a concise, day-by-day plan based on this outline:\n\n"
        "{learning_plan}\n\n"
        "Constraints: keep it practical, 60–90 minutes/day, list exact tasks and checkpoints."
    )
)

# 3) Build the LCEL chain:
#    - Take input {activity}
#    - Render learning_prompt -> llm -> parse to string => {learning_plan}
#    - Feed {learning_plan} into time_prompt -> llm -> parse to string (final answer)
seq_chain = (
    {
        "learning_plan": learning_prompt | llm | StrOutputParser()
    }
    | time_prompt
    | llm
    | StrOutputParser()
)

"""
We pipe components with LCEL:

- learning_prompt | llm | StrOutputParser() produces a string learning_plan.

- That string is injected into time_prompt, then passed to the LLM again and parsed to the final text.

- The dict {"learning_plan": ...} assigns the intermediate output to the variable name expected by time_prompt.
"""

if __name__ == "__main__":
    # Example run
    result = seq_chain.invoke({"activity": "play the harmonica"})
    print(result)
