# Tutorial 3: Creating Smart Question Templates

## 🎯 What You'll Learn

In this tutorial, you'll learn how to:

- Create reusable question templates instead of writing each question from scratch
- Use variables in your prompts to make them flexible and dynamic
- Build professional, consistent AI interactions
- Chain templates with AI models for powerful results

Think of this as creating a "fill-in-the-blanks" form that you can use over and over again!

## 🤔 Why Use Prompt Templates?

### Without Templates (Hard Way):
```python
# You have to write each question manually
response1 = llm.invoke("You are an AI assistant. Answer the question: What is the capital of France?")
response2 = llm.invoke("You are an AI assistant. Answer the question: What is the capital of Spain?")
response3 = llm.invoke("You are an AI assistant. Answer the question: What is the capital of Italy?")
```

### With Templates (Smart Way):
```python
# Write once, use many times with different questions
template = "You are an AI assistant. Answer the question: {question}"
response1 = template_chain.invoke({"question": "What is the capital of France?"})
response2 = template_chain.invoke({"question": "What is the capital of Spain?"})
response3 = template_chain.invoke({"question": "What is the capital of Italy?"})
```

## 🔍 Understanding the Code: Line by Line

Let's examine `script_03.py` step by step:

### Step 1: Importing Our Tools

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
```

**What's happening here?**

1. **`from langchain_openai import ChatOpenAI`**
   - Same as before - our connection to OpenAI's AI models

2. **`from langchain_core.prompts import PromptTemplate`**
   - This is the new star! `PromptTemplate` helps us create reusable question formats
   - Think of it as a template for writing professional emails - same structure, different content

3. **`from dotenv import load_dotenv` and `import os`**
   - Same as before - loading our secret API key

### Step 2: Setting Up Our Secrets

```python
# Load variables from .env file
load_dotenv()

# Get the API key from environment
api_key = os.getenv("OPENAI_API_KEY")
```

**What's happening here?**

- Same process as Tutorial 1 - loading our OpenAI API key securely
- No changes needed here!

### Step 3: Creating Our Question Template

```python
# Create a prompt template
template = "You are an artificial intelligence assistant. Answer the question clearly and concisely.\nQuestion: {question}"
prompt = PromptTemplate.from_template(template)
```

**What's happening here?**

1. **`template = "You are an artificial intelligence assistant..."`**
   - This is our reusable template string
   - Notice `{question}` - this is a placeholder (like a blank to fill in)
   - `\n` means "new line" (like pressing Enter)

2. **`prompt = PromptTemplate.from_template(template)`**
   - This converts our simple text into a smart `PromptTemplate` object
   - Now our template can be used with AI models

**💡 Breaking Down the Template:**

```
You are an artificial intelligence assistant.    ← Sets the AI's role
Answer the question clearly and concisely.      ← Gives instructions  
Question: {question}                            ← Placeholder for our actual question
```

### Step 4: Setting Up Our AI Model

```python
# Instantiate the LLM
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=api_key   # use api_key here for langchain_openai
)
```

**What's happening here?**

- Exactly the same as Tutorial 1
- Creating our AI assistant using GPT-4o

### Step 5: Creating a Chain (The Magic!)

```python
# Create a chain by piping prompt → llm
llm_chain = prompt | llm
```

**What's happening here?**

1. **`llm_chain = prompt | llm`**
   - This creates a "chain" - a sequence of operations
   - The `|` symbol means "pipe" or "then"
   - Read as: "Take the prompt, THEN send it to the LLM"

2. **What's a chain?**
   - It's like an assembly line in a factory
   - Step 1: Fill in the template with your question
   - Step 2: Send the completed prompt to the AI
   - Step 3: Get back the AI's response

### Step 6: Using Our Template

```python
# Question to send
question = "What is the capital of Malaysia?"

# Invoke the chain and print result
response = llm_chain.invoke({"question": question})
print(response.content)
```

**What's happening here?**

1. **`question = "What is the capital of Malaysia?"`**
   - This is the actual question we want to ask
   - It will replace `{question}` in our template

2. **`response = llm_chain.invoke({"question": question})`**
   - We're calling our chain with a dictionary: `{"question": question}`
   - The key `"question"` matches the `{question}` placeholder in our template
   - The chain fills in the template and sends it to the AI

3. **`print(response.content)`**
   - Display the AI's answer

## 🧠 What Happens Behind the Scenes?

### Step-by-Step Process:

1. **Template Filling**:
   ```
   Before: "You are an AI assistant. Answer: {question}"
   After:  "You are an AI assistant. Answer: What is the capital of Malaysia?"
   ```

2. **Sending to AI**:
   - The complete prompt gets sent to OpenAI's servers
   - The AI sees the full, professional prompt

3. **AI Response**:
   - The AI follows our instructions (be clear and concise)
   - Returns a focused answer

## 🚀 How to Run This Code

1. **Make sure your API key is set up**
   ```bash
   # Your .env file should contain:
   OPENAI_API_KEY=sk-your-actual-key-here
   ```

2. **Run the script**
   ```bash
   python script_03.py
   ```

3. **What you'll see**
   ```
   The capital of Malaysia is Kuala Lumpur.
   ```

## 🎓 Key Concepts You've Learned

### Prompt Templates
- **What**: Reusable text patterns with placeholders
- **Why**: Consistency, efficiency, professionalism
- **How**: Use `{variable_name}` for placeholders

### Chains
- **What**: Connected sequences of operations
- **Why**: Automate multi-step processes
- **How**: Use the `|` operator to connect components

### Template Variables
- **What**: Placeholders in curly braces like `{question}`
- **Why**: Makes templates flexible and reusable
- **How**: Pass values in a dictionary when calling the chain

### The Pipe Operator (`|`)
- **What**: Connects different components in a sequence
- **Why**: Creates smooth workflows
- **How**: `input | process1 | process2 | output`

## 🔧 Common Issues and Solutions

**Problem: "KeyError: 'question'"**
```python
# Wrong - missing the variable name
response = llm_chain.invoke("What is the capital of France?")

# Right - use a dictionary with the correct key
response = llm_chain.invoke({"question": "What is the capital of France?"})
```

**Problem: "Template variable not found"**
```python
# Wrong - variable names must match
template = "Answer this: {my_question}"
response = llm_chain.invoke({"question": "test"})  # Wrong key!

# Right - matching variable names
template = "Answer this: {question}"
response = llm_chain.invoke({"question": "test"})  # Correct key!
```

## 🎯 Try These Experiments

### 1. Multiple Variables Template
```python
template = "You are a {role}. Please {task} about {topic}."
prompt = PromptTemplate.from_template(template)

# Use it
response = llm_chain.invoke({
    "role": "geography teacher",
    "task": "explain",
    "topic": "mountain formation"
})
```

### 2. Different Question Types
```python
# Math questions
response = llm_chain.invoke({"question": "What is 15 × 23?"})

# Creative questions
response = llm_chain.invoke({"question": "Write a haiku about programming"})

# Factual questions
response = llm_chain.invoke({"question": "How do birds fly?"})
```

### 3. Personality Templates
```python
# Friendly assistant
template = "You are a friendly and enthusiastic assistant. {question}"

# Professional assistant  
template = "You are a professional consultant. Provide a detailed analysis: {question}"

# Creative assistant
template = "You are a creative writer. Answer imaginatively: {question}"
```

### 4. Structured Response Templates
```python
template = """Please answer the following question using this format:

Question: {question}

Answer: [Your answer here]
Explanation: [Brief explanation]
Source: [Where this information comes from]
"""
```

## 🌟 Advanced Template Features

### Input Variables Declaration
```python
# Explicit way to declare variables
prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="Context: {context}\nQuestion: {question}\nAnswer:"
)
```

### Template Validation
```python
# LangChain automatically checks if all variables are provided
# This will raise an error if you forget to provide a variable
```

## 📚 Template Best Practices

### 1. Be Specific with Instructions
```python
# Vague
template = "Answer: {question}"

# Better
template = "You are a helpful assistant. Answer the question clearly and concisely: {question}"
```

### 2. Use Descriptive Variable Names
```python
# Unclear
template = "Translate {a} to {b}"

# Clear
template = "Translate {text} from {source_language} to {target_language}"
```

### 3. Include Context When Needed
```python
template = """You are an expert in {domain}.
Context: {background_info}
Question: {user_question}
Please provide a detailed answer."""
```

## 🌟 What's Next?

Now that you understand prompt templates, you're ready to learn about:

- **Chat Templates** (Tutorial 4) - Multi-message conversations
- **Few-Shot Learning** (Tutorial 5) - Teaching AI through examples
- **Sequential Chains** (Tutorial 6) - Multiple AI steps in sequence

Congratulations! You can now create professional, reusable AI prompts! 🎉

## 💡 Real-World Applications

- **Customer Service**: Template for different types of support questions
- **Content Creation**: Templates for different types of writing
- **Education**: Templates for different subjects and question types
- **Research**: Templates for different kinds of analysis