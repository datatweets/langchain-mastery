# Tutorial 4: Having Real Conversations with Chat Templates

## 🎯 What You'll Learn

In this tutorial, you'll learn how to:

- Create multi-message conversations instead of single questions
- Use different types of messages (system, human, AI)
- Build context-aware AI assistants
- Create specialized AI personas through system messages

Think of this as teaching the AI to have a natural conversation with memory and context, like texting with a smart friend!

## 🤔 Single Message vs. Chat Conversations

### Single Message (What we did before):
```
You: "What colors are in France's flag?"
AI: "Blue, white, and red."
```

### Chat Conversation (What we're learning now):
```
System: "You are a geography expert that returns flag colors."
Human: "France"
AI: "blue, white, red"
Human: "Malaysia"  ← The AI remembers the context!
```

## 🗣️ Types of Messages in Chat

### 1. System Messages
- **Purpose**: Set the AI's personality and role
- **Who writes it**: You (the programmer)
- **Example**: "You are a helpful geography teacher"

### 2. Human Messages  
- **Purpose**: Questions or statements from the user
- **Who writes it**: The user
- **Example**: "What is the capital of France?"

### 3. AI Messages
- **Purpose**: Show examples of how the AI should respond
- **Who writes it**: You (as training examples)
- **Example**: Previous AI responses to guide future ones

## 🔍 Understanding the Code: Line by Line

Let's examine `script_04.py` step by step:

### Step 1: Importing the Chat Tools

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
```

**What's happening here?**

1. **`from langchain_core.prompts import ChatPromptTemplate`**
   - This is different from `PromptTemplate` we used before
   - `ChatPromptTemplate` is designed for multi-message conversations
   - It understands different types of messages (system, human, AI)

### Step 2: Setting Up Credentials

```python
# Load variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
```

**What's happening here?**

- Same as before - loading our OpenAI API key
- No changes needed here!

### Step 3: Creating the Chat Template

```python
# Define a chat prompt template with system, human, and AI messages
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a geography expert that returns the colors present in a country's flag."),
        ("human", "France"),
        ("ai", "blue, white, red"),
        ("human", "{country}")
    ]
)
```

**What's happening here?**

This is the most important part! Let's break it down:

1. **`ChatPromptTemplate.from_messages([...])`**
   - Creates a conversation template from a list of messages
   - Each message has a type and content

2. **Message Types and Content**:
   ```python
   ("system", "You are a geography expert...")  # Sets AI's role
   ("human", "France")                          # Example user input
   ("ai", "blue, white, red")                  # Example AI response
   ("human", "{country}")                       # New user input (variable)
   ```

3. **What this conversation looks like**:
   ```
   System: You are a geography expert that returns flag colors.
   Human: France
   AI: blue, white, red  
   Human: Malaysia  ← This is where {country} gets filled in
   ```

**💡 Why include the example?**

- The AI learns from the example: France → "blue, white, red"
- When asked about Malaysia, it knows to respond in the same format
- This is called "few-shot learning" (learning from examples)

### Step 4: Creating the AI Model

```python
# Create the LLM
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=api_key
)
```

**What's happening here?**

- Same as before - creating our GPT-4o assistant
- The AI model will receive the entire conversation

### Step 5: Building the Conversation Chain

```python
# Chain the prompt template to the LLM
llm_chain = prompt_template | llm
```

**What's happening here?**

- Same concept as Tutorial 3
- Now we're chaining a **chat template** with the AI
- The full conversation gets sent to the AI

### Step 6: Having the Conversation

```python
# Run the chain with a country input
country_name = "Malaysia"
response = llm_chain.invoke({"country": country_name})

# Print the result
print(response.content)
```

**What's happening here?**

1. **`country_name = "Malaysia"`**
   - This is what will replace `{country}` in our template

2. **`response = llm_chain.invoke({"country": country_name})`**
   - The entire conversation gets sent to the AI:
     - System message (sets the role)
     - Example human input (France)
     - Example AI response (blue, white, red)
     - New human input (Malaysia)

3. **The AI understands**:
   - I need to return flag colors (from system message)
   - I should respond like the example (from the France/blue,white,red example)
   - The user is asking about Malaysia (from the new input)

## 🧠 What Happens Behind the Scenes?

### The Full Conversation Sent to AI:
```
System: You are a geography expert that returns the colors present in a country's flag.
Human: France
Assistant: blue, white, red
Human: Malaysia
```

### AI's Response:
```
red, white, blue, yellow
```

The AI:
1. **Sees its role** (geography expert for flag colors)
2. **Learns from the example** (France → blue, white, red)
3. **Applies the pattern** to Malaysia
4. **Responds in the same format**

## 🚀 How to Run This Code

1. **Make sure your API key is set up**
   ```bash
   # Your .env file should contain:
   OPENAI_API_KEY=sk-your-actual-key-here
   ```

2. **Run the script**
   ```bash
   python script_04.py
   ```

3. **What you'll see**
   ```
   red, white, blue, yellow
   ```

## 🎓 Key Concepts You've Learned

### Chat Templates vs. Regular Templates

**Regular Template (Tutorial 3)**:
```python
template = "You are an AI. Answer: {question}"
# Single interaction
```

**Chat Template (This tutorial)**:
```python
ChatPromptTemplate.from_messages([
    ("system", "You are an expert..."),
    ("human", "Example input"),
    ("ai", "Example output"),
    ("human", "{new_input}")
])
# Multi-message conversation with context
```

### Message Types
- **system**: Sets personality, role, and behavior
- **human**: User inputs and questions
- **ai**: AI responses (used as examples)

### Conversation Context
- The AI sees the entire conversation history
- Previous messages provide context for new responses
- Examples teach the AI how to respond

## 🔧 Common Issues and Solutions

**Problem: AI responds with too much text**
```python
# Solution: Be more specific in the system message
("system", "You are a geography expert. Return ONLY the colors in the flag, separated by commas.")
```

**Problem: AI doesn't follow the format**
```python
# Solution: Add more examples
[
    ("system", "Return flag colors only."),
    ("human", "France"),
    ("ai", "blue, white, red"),
    ("human", "Italy"), 
    ("ai", "green, white, red"),  # More examples help!
    ("human", "{country}")
]
```

**Problem: Variable name mismatch**
```python
# Wrong - variable names must match
("human", "{country}")
response = llm_chain.invoke({"nation": "Japan"})  # Wrong key!

# Right - matching variable names  
("human", "{country}")
response = llm_chain.invoke({"country": "Japan"})  # Correct key!
```

## 🎯 Try These Experiments

### 1. Different Types of Experts
```python
# Math expert
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a math tutor. Solve problems step by step."),
    ("human", "What is 15 + 27?"),
    ("ai", "15 + 27 = 42"),
    ("human", "{math_problem}")
])
```

### 2. Creative Writing Assistant
```python
# Story writer
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a creative writer. Write short, engaging stories."),
    ("human", "dragons"),
    ("ai", "Once upon a time, a small dragon learned to breathe flowers instead of fire."),
    ("human", "{topic}")
])
```

### 3. Multiple Variables
```python
# Language translator with examples
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Translate from {source_lang} to {target_lang}."),
    ("human", "hello"),
    ("ai", "hola"),
    ("human", "{word}")
])

# Use it
response = llm_chain.invoke({
    "source_lang": "English",
    "target_lang": "Spanish", 
    "word": "goodbye"
})
```

### 4. Longer Conversations
```python
# Customer service bot
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer service representative."),
    ("human", "I have a problem with my order."),
    ("ai", "I'd be happy to help! Can you please provide your order number?"),
    ("human", "Order #12345"),
    ("ai", "Thank you! I can see your order. What specific issue are you experiencing?"),
    ("human", "{customer_issue}")
])
```

## 🌟 Advanced Chat Features

### Variable in System Messages
```python
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} expert. Be {style} in your responses."),
    ("human", "{question}")
])

response = llm_chain.invoke({
    "role": "cooking",
    "style": "enthusiastic", 
    "question": "How do I make pasta?"
})
```

### Multiple Examples for Better Learning
```python
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a rhyme expert. Find words that rhyme."),
    ("human", "cat"),
    ("ai", "bat, hat, mat, rat"),
    ("human", "dog"), 
    ("ai", "log, frog, hog"),
    ("human", "tree"),
    ("ai", "bee, free, sea, key"),
    ("human", "{word}")  # Now the AI has learned the pattern well!
])
```

## 📚 Chat Template Best Practices

### 1. Clear System Messages
```python
# Vague
("system", "You help with questions.")

# Specific  
("system", "You are a professional nutritionist. Provide evidence-based advice about healthy eating.")
```

### 2. Representative Examples
```python
# Use examples that show the exact format you want
("human", "apple"),
("ai", "Fruit: Rich in fiber and vitamin C. Calories: ~95 per medium apple.")
```

### 3. Consistent Formatting
```python
# Keep the same format throughout your examples
("human", "input1"), ("ai", "formatted response 1"),
("human", "input2"), ("ai", "formatted response 2"),
```

## 🌟 What's Next?

Now that you understand chat templates and conversations, you're ready to learn about:

- **Few-Shot Learning** (Tutorial 5) - Teaching AI through multiple examples
- **Sequential Chains** (Tutorial 6) - Connecting multiple AI steps
- **AI Agents** (Tutorial 7) - AI that can use tools and search

Congratulations! You can now create AI assistants with personality and context! 🎉

## 💡 Real-World Applications

- **Customer Support**: Context-aware help systems
- **Educational Tutors**: Personalized learning assistants  
- **Creative Tools**: Writing assistants with specific styles
- **Domain Experts**: Specialized knowledge systems (medical, legal, technical)