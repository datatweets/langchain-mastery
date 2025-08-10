# Tutorial 6: Building AI Workflows with Sequential Chains

## 🎯 What You'll Learn

In this tutorial, you'll learn how to:

- Connect multiple AI steps into a workflow (chain)
- Pass information from one AI step to the next
- Create complex multi-step AI processes
- Build a practical learning plan generator

Think of this as creating an assembly line where each AI step improves and refines the work of the previous step!

## 🤔 Single Step vs. Multi-Step AI Processing

### Single Step (What we did before):
```
You: "Help me learn harmonica"
AI: "Here's a comprehensive guide to learning harmonica..." (gives everything at once)
```

### Multi-Step Chain (What we're learning now):
```
Step 1: "Create a learning plan for harmonica"
AI: Creates detailed learning plan

Step 2: "Compress this plan into a 1-week schedule"
AI: Takes the first plan and creates a focused weekly schedule
```

**The Power**: Each step builds upon the previous one, creating more refined and targeted results!

## 🔍 Understanding the Code: Line by Line

Let's examine `script_06.py` step by step:

### Step 1: Importing Our Tools

```python
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
```

**What's happening here?**

1. **`StrOutputParser`** - This is new!
   - Converts AI responses into clean text strings
   - Removes extra formatting and metadata
   - Makes output ready for the next step in the chain

### Step 2: Loading Environment Variables

```python
# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

**What's happening here?**

- Same as previous tutorials - loading our OpenAI API key
- No changes needed here!

### Step 3: Creating the AI Model

```python
# LLM (gpt-4o)
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=api_key,
    temperature=0.3,
)
```

**What's happening here?**

1. **`temperature=0.3`** - This is new!
   - Controls creativity vs. consistency
   - 0.0 = Very consistent, factual responses
   - 1.0 = Very creative, unpredictable responses
   - 0.3 = Good balance for learning plans

### Step 4: Creating the First Prompt Template

```python
# 1) Prompt to draft a step-by-step learning plan for the activity
learning_prompt = PromptTemplate(
    input_variables=["activity"],
    template="I want to learn how to {activity}. "
             "Suggest a clear, step-by-step learning plan with milestones and practice tasks."
)
```

**What's happening here?**

1. **`learning_prompt`** - Our first step in the chain
2. **`input_variables=["activity"]`** - Expects an "activity" parameter
3. **Template purpose**: Creates a comprehensive learning plan
4. **Example**: If activity = "harmonica", it creates a full learning roadmap

### Step 5: Creating the Second Prompt Template

```python
# 2) Prompt to compress that plan into a concise one-week schedule
time_prompt = PromptTemplate(
    input_variables=["learning_plan"],
    template=(
        "I only have one week. Create a concise, day-by-day plan based on this outline:\n\n"
        "{learning_plan}\n\n"
        "Constraints: keep it practical, 60–90 minutes/day, list exact tasks and checkpoints."
    )
)
```

**What's happening here?**

1. **`time_prompt`** - Our second step in the chain
2. **`input_variables=["learning_plan"]`** - Takes output from step 1
3. **Template purpose**: Compresses the full plan into a practical 1-week schedule
4. **Key constraints**: 60-90 minutes/day, specific daily tasks

### Step 6: Building the Sequential Chain (The Magic!)

```python
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
```

**What's happening here?** (This is the most complex part!)

1. **LCEL (LangChain Expression Language)** - A way to build chains
2. **Let's break down each part**:

   ```python
   # Step 1: Create the learning plan
   learning_prompt | llm | StrOutputParser()
   ```
   - Takes `{activity}` → creates learning plan prompt → sends to AI → gets clean text

   ```python
   # Step 2: Wrap it in a dictionary
   {
       "learning_plan": learning_prompt | llm | StrOutputParser()
   }
   ```
   - Assigns the result to variable name "learning_plan"

   ```python
   # Step 3: Chain it with the time prompt
   | time_prompt
   | llm
   | StrOutputParser()
   ```
   - Takes the learning_plan → creates time-based prompt → sends to AI → gets final result

3. **The complete flow**:
   ```
   Input: {activity: "harmonica"}
   ↓
   Step 1: learning_prompt + AI → detailed learning plan
   ↓
   Step 2: time_prompt + AI → 1-week schedule
   ↓
   Output: Practical daily schedule
   ```

### Step 7: Understanding the Chain Comments

```python
"""
We pipe components with LCEL:

- learning_prompt | llm | StrOutputParser() produces a string learning_plan.

- That string is injected into time_prompt, then passed to the LLM again and parsed to the final text.

- The dict {"learning_plan": ...} assigns the intermediate output to the variable name expected by time_prompt.
"""
```

**What's happening here?**

- This is documentation explaining the complex chain
- **Piping**: Using `|` to connect components
- **Variable assignment**: The dictionary creates a variable for the next step
- **Two AI calls**: The chain calls the AI twice, each with different prompts

### Step 8: Running the Chain

```python
if __name__ == "__main__":
    # Example run
    result = seq_chain.invoke({"activity": "play the harmonica"})
    print(result)
```

**What's happening here?**

1. **`seq_chain.invoke({"activity": "play the harmonica"})`**
   - Starts the chain with the activity "play the harmonica"
   - Runs through both AI steps automatically
   - Returns the final 1-week schedule

2. **What happens internally**:
   - AI creates a comprehensive harmonica learning plan
   - AI takes that plan and creates a focused 1-week version
   - You get the practical daily schedule

## 🧠 What Happens Behind the Scenes?

### Step-by-Step Process:

1. **Input**: `{"activity": "play the harmonica"}`

2. **First AI Call**:
   ```
   Prompt: "I want to learn how to play the harmonica. Suggest a clear, step-by-step learning plan with milestones and practice tasks."
   
   AI Response: "Week 1: Learn basic breathing techniques and single notes...
   Week 2: Practice simple melodies...
   Week 3: Learn bending techniques..." (detailed plan)
   ```

3. **Second AI Call**:
   ```
   Prompt: "I only have one week. Create a concise, day-by-day plan based on this outline:
   
   [Full learning plan from step 2]
   
   Constraints: keep it practical, 60–90 minutes/day, list exact tasks and checkpoints."
   
   AI Response: "Day 1: Practice breathing (20 min) + single notes (40 min)
   Day 2: Review single notes (30 min) + simple scales (60 min)..."
   ```

4. **Final Output**: Practical 1-week daily schedule

## 🚀 How to Run This Code

1. **Make sure your API key is set up**
   ```bash
   # Your .env file should contain:
   OPENAI_API_KEY=sk-your-actual-key-here
   ```

2. **Run the script**
   ```bash
   python script_06.py
   ```

3. **What you'll see**
   ```
   Day 1 (60-90 min):
   • Morning (30 min): Learn proper harmonica holding technique
   • Afternoon (30-45 min): Practice single note breathing on holes 4, 5, 6
   • Evening (15 min): Listen to simple harmonica songs for ear training
   
   Day 2 (60-90 min):
   • Morning (45 min): Practice clean single notes across all holes
   • Afternoon (30 min): Learn basic major scale pattern
   • Evening (15 min): Record yourself playing single notes
   ...
   ```

## 🎓 Key Concepts You've Learned

### Sequential Chains
- **What**: AI workflows where each step builds on the previous one
- **Why**: Creates more refined, focused results than single-step processing
- **How**: Use LCEL to connect prompts, AI calls, and parsers

### Output Parsers
- **What**: Tools that clean up AI responses
- **Why**: Makes output ready for the next step in the chain
- **How**: `StrOutputParser()` converts to clean text

### LCEL (LangChain Expression Language)
- **What**: A syntax for building complex AI workflows
- **Why**: Makes chains readable and maintainable
- **How**: Use `|` to pipe components together

### Temperature Setting
- **What**: Controls AI creativity vs. consistency
- **Why**: Different tasks need different levels of creativity
- **How**: 0.0-1.0 scale, where 0.3 is good for structured tasks

### Variable Passing Between Steps
- **What**: Using dictionaries to pass data between chain steps
- **Why**: Allows complex multi-step processing
- **How**: `{"variable_name": first_step} | second_step`

## 🔧 Common Issues and Solutions

**Problem: "Variable not found in template"**
```python
# Wrong - variable names must match
{"learning_plan": step1} | PromptTemplate(template="Use this {plan}")  # Mismatch!

# Right - matching variable names
{"learning_plan": step1} | PromptTemplate(template="Use this {learning_plan}")  # Match!
```

**Problem: "Chain is too slow"**
```python
# Solution: Each step calls the AI, so 2 steps = 2 API calls
# This is normal - complex chains take more time
# You can optimize by using faster models for some steps
```

**Problem: "Output format is inconsistent"**
```python
# Solution: Add more specific instructions to your prompts
template = "Create a day-by-day plan. Format each day as 'Day X: [tasks]'"
```

## 🎯 Try These Experiments

### 1. Three-Step Chain
```python
# Step 1: Create learning plan
# Step 2: Create 1-week schedule  
# Step 3: Create today's specific tasks

today_prompt = PromptTemplate(
    input_variables=["weekly_plan"],
    template="Based on this weekly plan:\n{weekly_plan}\n\nWhat exactly should I do today? List specific 15-minute tasks."
)

three_step_chain = (
    {"learning_plan": learning_prompt | llm | StrOutputParser()}
    | {"weekly_plan": time_prompt | llm | StrOutputParser()}
    | today_prompt
    | llm
    | StrOutputParser()
)
```

### 2. Different Learning Topics
```python
# Try different activities
result = seq_chain.invoke({"activity": "cooking Italian food"})
result = seq_chain.invoke({"activity": "learning Spanish"})
result = seq_chain.invoke({"activity": "playing chess"})
```

### 3. Research and Summarize Chain
```python
research_prompt = PromptTemplate(
    input_variables=["topic"],
    template="List the most important facts and concepts about {topic}. Be comprehensive."
)

summary_prompt = PromptTemplate(
    input_variables=["research"],
    template="Summarize this research into 3 key takeaways:\n{research}"
)

research_chain = (
    {"research": research_prompt | llm | StrOutputParser()}
    | summary_prompt
    | llm  
    | StrOutputParser()
)
```

### 4. Writing Assistant Chain
```python
outline_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Create an outline for an article about {topic}."
)

writing_prompt = PromptTemplate(
    input_variables=["outline"],
    template="Write a 500-word article based on this outline:\n{outline}"
)

writing_chain = (
    {"outline": outline_prompt | llm | StrOutputParser()}
    | writing_prompt
    | llm
    | StrOutputParser()
)
```

## 🌟 Advanced Chain Techniques

### Parallel Processing
```python
# Run multiple steps in parallel, then combine
parallel_chain = (
    {
        "advantages": advantages_prompt | llm | StrOutputParser(),
        "disadvantages": disadvantages_prompt | llm | StrOutputParser()
    }
    | comparison_prompt
    | llm
    | StrOutputParser()
)
```

### Conditional Chains
```python
# Different paths based on input
def route_chain(input_data):
    if "beginner" in input_data["level"]:
        return beginner_chain
    else:
        return advanced_chain
```

### Error Handling
```python
# Add try/catch for robust chains
try:
    result = seq_chain.invoke({"activity": activity})
except Exception as e:
    print(f"Chain failed: {e}")
    # Fallback logic here
```

## 📚 Chain Design Best Practices

### 1. Clear Variable Names
```python
# Good - descriptive names
{"detailed_plan": step1, "weekly_schedule": step2}

# Bad - unclear names  
{"output1": step1, "result": step2}
```

### 2. Specific Prompts
```python
# Good - specific instructions
template = "Create exactly 7 daily tasks, formatted as 'Day X: [task]'"

# Bad - vague instructions
template = "Make a plan"
```

### 3. Appropriate Chain Length
- **2-3 steps**: Most common, good balance
- **4-5 steps**: Complex workflows
- **6+ steps**: Usually too complex, consider breaking up

## 🌟 What's Next?

Now that you understand sequential chains, you're ready to learn about:

- **AI Agents** (Tutorial 7) - AI that can use tools and make decisions
- **RAG Systems** (Tutorial 8) - AI that can read and analyze documents
- **Advanced RAG** (Tutorial 9) - Sophisticated document analysis systems

Congratulations! You can now create complex multi-step AI workflows! 🎉

## 💡 Real-World Applications

- **Content Creation**: Research → Outline → Write → Edit
- **Learning Systems**: Assess → Plan → Schedule → Track
- **Business Analysis**: Gather Data → Analyze → Summarize → Recommend
- **Code Development**: Plan → Code → Test → Document