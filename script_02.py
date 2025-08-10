# Import the class for defining Hugging Face pipelines
from langchain_huggingface import HuggingFacePipeline

# Define the LLM from the Hugging Face model ID
llm = HuggingFacePipeline.from_model_id(
    # model_id="crumb/nano-mistral",
    model_id="microsoft/DialoGPT-medium",
    # model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 20}
)

prompt = "URL is"

# Invoke the model
response = llm.invoke(prompt)
print(response)