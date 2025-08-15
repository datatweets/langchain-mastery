import os
from dotenv import load_dotenv
from langsmith import Client

# Load environment variables from .env file
load_dotenv()

# Get the LangSmith API key from environment variables
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if not langsmith_api_key:
    print("❌ No LANGSMITH_API_KEY found in .env file.")
    exit(1)

# Initialize LangSmith client
client = Client(api_key=langsmith_api_key)

# Test inputs - questions for our dataset
test_inputs = [
    "What is LangSmith?",
    "What is LangServe?", 
    "How could I benchmark RAG on tables?",
    "What was exciting about LangChain's first birthday?",
    "What features were released for LangChain on August 7th?",
    "What is a conversational retrieval agent?"
]

# Dataset configuration
dataset_name = "langsmith-demo-dataset-v1"
dataset_description = "LangChain Blog Test Questions"

try:
    # Create the dataset
    print(f"🔄 Creating dataset: {dataset_name}")
    dataset = client.create_dataset(
        dataset_name=dataset_name, 
        description=dataset_description
    )
    print(f"✅ Dataset created successfully with ID: {dataset.id}")
    
    # Add examples to the dataset
    print(f"🔄 Adding {len(test_inputs)} examples to the dataset...")
    
    for i, input_question in enumerate(test_inputs, 1):
        print(f"  Adding example {i}/{len(test_inputs)}: {input_question[:50]}...")
        
        # Create example with input and expected output structure
        # Note: In a real scenario, you would have actual expected answers
        # For demo purposes, we're using a placeholder structure
        client.create_example(
            inputs={"question": input_question},
            outputs={"answer": f"Expected answer for: {input_question}"},
            dataset_id=dataset.id
        )
    
    print(f"✅ Successfully added all examples to dataset '{dataset_name}'")
    print(f"📊 Dataset URL: https://smith.langchain.com/datasets/{dataset.id}")
    
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"⚠️  Dataset '{dataset_name}' already exists. Trying to retrieve it...")
        try:
            # Get existing dataset
            datasets = client.list_datasets(dataset_name=dataset_name)
            if datasets:
                dataset = list(datasets)[0]
                print(f"✅ Found existing dataset with ID: {dataset.id}")
                print(f"📊 Dataset URL: https://smith.langchain.com/datasets/{dataset.id}")
            else:
                print("❌ Could not retrieve existing dataset")
        except Exception as retrieve_error:
            print(f"❌ Error retrieving dataset: {retrieve_error}")
    else:
        print(f"❌ Error creating dataset: {e}")

# Optional: List all datasets to verify
try:
    print("\n📋 Current datasets in your LangSmith account:")
    datasets = client.list_datasets()
    for dataset in datasets:
        print(f"  - {dataset.name} (ID: {dataset.id})")
except Exception as e:
    print(f"❌ Error listing datasets: {e}")