# Tutorial 5: Teaching AI Through Examples (Few-Shot Learning)

## 🎯 What You'll Learn

In this tutorial, you'll learn how to:

- Teach AI models by showing them multiple examples
- Create consistent, predictable AI responses
- Use few-shot learning to guide AI behavior
- Build AI systems that follow specific patterns and formats

Think of this as showing a student several solved problems before asking them to solve a new one!

## 🤔 What is Few-Shot Learning?

### Zero-Shot (No Examples):
```
AI: "Answer questions about GitHub users."
Human: "How many repositories does Lina have?"
AI: "I don't have access to real GitHub data..." ❌
```

### Few-Shot (With Examples):
```
AI sees these examples first:
- Q: "How many repositories does Lina have?" → A: "42"
- Q: "How many stars has Lina received?" → A: "1,580" 
- Q: "What language does Lina use most?" → A: "Python"

Then when asked: "What is Lina's primary programming language?"
AI: "Python" ✅ (Follows the pattern!)
```

**The Magic**: The AI learns the pattern and style from examples, then applies it to new questions!

## 🔍 Understanding the Code: Line by Line

Let's examine `script_05.py` step by step:

### Step 1: Importing Our Tools

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from dotenv import load_dotenv
import os
```

**What's happening here?**

1. **`FewShotPromptTemplate`** - This is the new star!
   - Different from `PromptTemplate` we used before
   - Specifically designed to handle multiple examples
   - Automatically formats examples in a consistent way

### Step 2: Loading API Credentials

```python
# Load env vars
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

**What's happening here?**

- Same as previous tutorials - loading our OpenAI API key
- No changes needed here!

### Step 3: Creating Our Training Examples

```python
# --- Few-shot examples about a GitHub user (Lina) ---
examples = [
    {"question": "How many public repositories does Lina have on GitHub?", "answer": "42"},
    {"question": "How many total stars has Lina received across repositories?", "answer": "1,580"},
    {"question": "Which language does Lina use most on GitHub?", "answer": "Python"},
]
```

**What's happening here?**

1. **`examples = [...]`**
   - This is a list of dictionaries (like a database of examples)
   - Each dictionary has a "question" and an "answer"

2. **Why these specific examples?**
   - They're about the same person (Lina) - creates consistency
   - They follow the same format - short, direct answers
   - They cover different types of GitHub information

3. **Structure of each example**:
   ```python
   {
       "question": "A specific question about Lina",
       "answer": "A short, direct answer"
   }
   ```

**💡 Why use fake data?**
- This is for demonstration purposes
- In real applications, you'd use real data or examples
- The AI learns the response style, not the specific facts

### Step 4: Creating the Example Format Template

```python
# How each example is rendered
example_prompt = PromptTemplate.from_template(
    "Question: {question}\nAnswer: {answer}"
)
```

**What's happening here?**

1. **`example_prompt`** defines how each example will look
2. **The template**: `"Question: {question}\nAnswer: {answer}"`
   - Creates a consistent format for every example
   - `\n` means "new line"

3. **What this creates for each example**:
   ```
   Question: How many repositories does Lina have?
   Answer: 42
   
   Question: How many stars has Lina received?
   Answer: 1,580
   ```

### Step 5: Building the Few-Shot Template

```python
# Build the few-shot prompt
prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}\nAnswer:",
    input_variables=["input"],
)
```

**What's happening here?**

1. **`FewShotPromptTemplate(...)`** - Creates our special template

2. **`examples=examples`** - Uses our training examples

3. **`example_prompt=example_prompt`** - Uses our formatting template

4. **`suffix="Question: {input}\nAnswer:"`**
   - This comes AFTER all the examples
   - `{input}` is where the new question goes
   - Ends with "Answer:" to prompt the AI to respond

5. **`input_variables=["input"]`**
   - Tells the template which variables to expect

### Step 6: Creating the AI Model

```python
# LLM (works with gpt-4o)
llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
```

**What's happening here?**

- Same as before - creating our GPT-4o assistant
- The AI will receive all examples plus the new question

### Step 7: Building the Chain

```python
# Chain prompt -> llm
chain = prompt_template | llm
```

**What's happening here?**

- Same concept as previous tutorials
- Now we're chaining a **few-shot template** with the AI

### Step 8: Asking a New Question

```python
# Ask a new question (model will mimic the examples' style)
user_question = "What is Lina's primary programming language on GitHub?"
response = chain.invoke({"input": user_question})
```

**What's happening here?**

1. **`user_question`** - Our new question (not in the examples)
2. **`chain.invoke({"input": user_question})`** - Sends everything to AI
3. **What the AI receives**:
   ```
   Question: How many repositories does Lina have?
   Answer: 42
   
   Question: How many stars has Lina received?
   Answer: 1,580
   
   Question: Which language does Lina use most?
   Answer: Python
   
   Question: What is Lina's primary programming language?
   Answer: [AI fills this in]
   ```

### Step 9: Seeing the Final Prompt (Optional)

```python
# (Optional) inspect the final prompt that was sent
formatted = prompt_template.invoke({"input": user_question})
print("\n--- Formatted Prompt ---\n")
print(formatted.to_string(), response.content)
```

**What's happening here?**

1. **`formatted = prompt_template.invoke(...)`** - Shows us the complete prompt
2. **`formatted.to_string()`** - Converts it to readable text
3. **This helps us see exactly what the AI received**

## 🧠 What the AI Actually Sees

When you run this code, the AI receives this complete prompt:

```
Question: How many public repositories does Lina have on GitHub?
Answer: 42

Question: How many total stars has Lina received across repositories?
Answer: 1,580

Question: Which language does Lina use most on GitHub?
Answer: Python

Question: What is Lina's primary programming language on GitHub?
Answer:
```

**The AI thinks**: 
- "I see a pattern here - short, direct answers about Lina"
- "The last question is similar to the Python question"
- "I should respond in the same style: just 'Python'"

## 🚀 How to Run This Code

1. **Make sure your API key is set up**
   ```bash
   # Your .env file should contain:
   OPENAI_API_KEY=sk-your-actual-key-here
   ```

2. **Run the script**
   ```bash
   python script_05.py
   ```

3. **What you'll see**
   ```
   --- Formatted Prompt ---
   
   Question: How many public repositories does Lina have on GitHub?
   Answer: 42
   
   Question: How many total stars has Lina received across repositories?
   Answer: 1,580
   
   Question: Which language does Lina use most on GitHub?
   Answer: Python
   
   Question: What is Lina's primary programming language on GitHub?
   Answer: Python
   ```

## 🎓 Key Concepts You've Learned

### Few-Shot Learning
- **What**: Teaching AI through examples before asking new questions
- **Why**: Creates consistent, predictable responses
- **How**: Show multiple examples, then ask a similar question

### Example Structure
- **Consistent format**: All examples follow the same pattern
- **Relevant examples**: Examples should be similar to expected use cases
- **Quality over quantity**: Better to have 3 good examples than 10 poor ones

### Template Components
- **Examples**: Your training data
- **Example format**: How each example is displayed  
- **Suffix**: The final part where new questions go
- **Input variables**: What the user will provide

## 🔧 Common Issues and Solutions

**Problem: AI doesn't follow the pattern**
```python
# Solution: Make sure examples are consistent
examples = [
    {"question": "What is the capital?", "answer": "Paris"},      # Inconsistent
    {"question": "Population of France?", "answer": "67 million"}, # Different format
]

# Better: Consistent format
examples = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the capital of Italy?", "answer": "Rome"},
]
```

**Problem: AI gives long answers despite short examples**
```python
# Solution: Add more examples with short answers
examples = [
    {"question": "Color of sun?", "answer": "Yellow"},
    {"question": "Color of grass?", "answer": "Green"},
    {"question": "Color of sky?", "answer": "Blue"},
    # More examples reinforce the pattern
]
```

**Problem: Variable name mismatch**
```python
# Wrong
suffix="Question: {input}\nAnswer:"
response = chain.invoke({"question": "test"})  # Wrong key!

# Right  
suffix="Question: {input}\nAnswer:"
response = chain.invoke({"input": "test"})     # Correct key!
```

## 🎯 Try These Experiments

### 1. Math Problem Solver
```python
examples = [
    {"problem": "2 + 3", "solution": "5"},
    {"problem": "10 - 4", "solution": "6"},
    {"problem": "3 × 7", "solution": "21"},
]

example_prompt = PromptTemplate.from_template("Problem: {problem}\nSolution: {solution}")

prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Problem: {problem}\nSolution:",
    input_variables=["problem"]
)
```

### 2. Language Translator
```python
examples = [
    {"english": "hello", "spanish": "hola"},
    {"english": "goodbye", "spanish": "adiós"},
    {"english": "thank you", "spanish": "gracias"},
]

example_prompt = PromptTemplate.from_template("English: {english}\nSpanish: {spanish}")
```

### 3. Code Explainer
```python
examples = [
    {
        "code": "print('hello')",
        "explanation": "Prints the word 'hello' to the screen"
    },
    {
        "code": "x = 5",
        "explanation": "Assigns the value 5 to variable x"
    },
]

example_prompt = PromptTemplate.from_template("Code: {code}\nExplanation: {explanation}")
```

### 4. Creative Writing Prompts
```python
examples = [
    {"theme": "ocean", "story": "The lighthouse keeper discovered a message in a bottle."},
    {"theme": "forest", "story": "Ancient trees whispered secrets to those who listened."},
    {"theme": "city", "story": "Street lights flickered as midnight secrets emerged."},
]
```

## 🌟 Advanced Few-Shot Techniques

### Dynamic Example Selection
```python
# Use different examples based on the question type
def select_examples(question):
    if "math" in question.lower():
        return math_examples
    elif "history" in question.lower():
        return history_examples
    else:
        return general_examples
```

### Example with Multiple Variables
```python
example_prompt = PromptTemplate.from_template(
    "Context: {context}\nQuestion: {question}\nAnswer: {answer}"
)

examples = [
    {
        "context": "Python programming",
        "question": "How do you print text?",
        "answer": "Use print('text')"
    }
]
```

## 📚 Few-Shot Best Practices

### 1. Quality Examples
```python
# Good: Specific, clear, consistent
{"question": "Capital of Japan?", "answer": "Tokyo"}

# Bad: Vague, inconsistent
{"question": "Japan stuff?", "answer": "Tokyo is the capital and it's very crowded and..."}
```

### 2. Diverse but Related Examples
```python
# Cover different variations of the same task
examples = [
    {"input": "happy", "output": "😊"},      # Emotion
    {"input": "sad", "output": "😢"},        # Emotion  
    {"input": "cat", "output": "🐱"},        # Animal
    {"input": "car", "output": "🚗"},        # Object
]
```

### 3. Appropriate Number of Examples
- **2-3 examples**: Simple tasks
- **3-5 examples**: Moderate complexity
- **5-10 examples**: Complex patterns
- **More than 10**: Usually overkill and expensive

## 🌟 What's Next?

Now that you understand few-shot learning, you're ready to learn about:

- **Sequential Chains** (Tutorial 6) - Connecting multiple AI steps
- **AI Agents** (Tutorial 7) - AI that can use tools and make decisions
- **RAG Systems** (Tutorial 8) - AI that can read and analyze documents

Congratulations! You can now train AI models through examples! 🎉

## 💡 Real-World Applications

- **Customer Service**: Training chatbots with example conversations
- **Content Creation**: Teaching AI to write in specific styles
- **Data Processing**: Showing AI how to format and clean data
- **Code Generation**: Teaching AI programming patterns through examples