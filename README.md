# LangChain Mastery: Complete Tutorial Collection

A comprehensive collection of 14 Python scripts with detailed tutorials demonstrating the complete LangChain ecosystem, from basic LLM interactions to sophisticated multi-tool AI assistants with persistent memory and graph-based workflows.

[![GitHub](https://img.shields.io/github/license/datatweets/langchain-mastery)](https://github.com/datatweets/langchain-mastery/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/datatweets/langchain-mastery)](https://github.com/datatweets/langchain-mastery/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/datatweets/langchain-mastery)](https://github.com/datatweets/langchain-mastery/network)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## 🎯 What Makes This Special

This repository provides a **complete learning journey** with:
- **14 Progressive Scripts**: From beginner to advanced concepts
- **14 Detailed Tutorials**: Every line explained in plain English
- **Production-Ready Examples**: Real-world patterns and best practices
- **Visual Workflows**: Automatic diagram generation for complex systems
- **Memory Management**: Persistent conversations across sessions

## 📚 Complete Script & Tutorial Collection

### 🟢 **Beginner Level (Scripts 1-6)**

| Script | Tutorial | Topic | Key Learning |
|--------|----------|--------|---------------|
| `script_01.py` | `tutorial_01.md` | **Basic OpenAI Integration** | API calls, environment setup, first AI conversation |
| `script_02.py` | `tutorial_02.md` | **Hugging Face Local Models** | Offline AI, model downloading, local inference |
| `script_03.py` | `tutorial_03.md` | **Prompt Templates** | Variable substitution, reusable prompts, structured input |
| `script_04.py` | `tutorial_04.md` | **Chat Templates** | Multi-message conversations, roles, context management |
| `script_05.py` | `tutorial_05.md` | **Few-Shot Learning** | Training through examples, pattern recognition, smart prompting |
| `script_06.py` | `tutorial_06.md` | **Sequential Chains** | Multi-step workflows, chaining operations, complex processing |

### 🟡 **Intermediate Level (Scripts 7-9)**

| Script | Tutorial | Topic | Key Learning |
|--------|----------|--------|---------------|
| `script_07.py` | `tutorial_07.md` | **AI Agents with Tools** | ReAct framework, Wikipedia integration, intelligent tool usage |
| `script_08.py` | `tutorial_08.md` | **Basic RAG System** | Document processing, vector databases, retrieval-augmented generation |
| `script_09.py` | `tutorial_09.md` | **Advanced RAG System** | Smart retrieval, persistent storage, document intelligence |

### 🔴 **Advanced Level (Scripts 10-14)**

| Script | Tutorial | Topic | Key Learning |
|--------|----------|--------|---------------|
| `script_10.py` | `tutorial_10.md` | **Custom Tools** | @tool decorator, mathematical functions, tool creation patterns |
| `script_11.py` | `tutorial_11.md` | **Conversation Memory** | Chat history, context awareness, persistent conversations |
| `script_12.py` | `tutorial_12.md` | **LangGraph Basics** | Graph workflows, visual AI, state management, streaming responses |
| `script_13.py` | `tutorial_13.md` | **Smart Research Assistant** | Wikipedia + LangGraph + Memory, intelligent routing, research workflows |
| `script_14.py` | `tutorial_14.md` | **Multi-Tool AI Assistant** | Ultimate AI Swiss Army knife, multiple tool types, production-ready system |

## 🚀 Quick Start

### 1. Clone the Repository
```bash
# Clone the repository
git clone https://github.com/datatweets/langchain-mastery.git

# Navigate to the project directory
cd langchain-mastery
```

### 2. Prerequisites
- **Python 3.11+** (recommended)
- **OpenAI API Key** ([Get one here](https://platform.openai.com/account/api-keys))

### 3. Setup Environment
```bash
# Create and activate virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 4. Configure API Keys
```bash
# Copy the sample environment file
cp .env-sample .env

# Edit .env file and add your OpenAI API key:
# OPENAI_API_KEY=sk-your-actual-key-here
```

### 5. Start Learning!
```bash
# Begin with the basics
python script_01.py

# Read the tutorial
cat tutorial_01.md

# Progress through the series
python script_02.py
# ... and so on
```

## 🎓 Learning Path Recommendations

### **For Complete Beginners:**
```
script_01.py → script_03.py → script_04.py → script_06.py → script_07.py
(Basic AI → Templates → Conversations → Chains → Agents)
```

### **For Developers with AI Experience:**
```
script_07.py → script_08.py → script_12.py → script_13.py → script_14.py
(Agents → RAG → LangGraph → Research → Multi-tool)
```

### **For RAG/Document Processing Focus:**
```
script_08.py → script_09.py → script_13.py
(Basic RAG → Advanced RAG → Research Assistant)
```

### **For Conversational AI Focus:**
```
script_04.py → script_11.py → script_12.py → script_14.py
(Chat Templates → Memory → LangGraph → Multi-tool)
```

## 🔧 Key Technologies Covered

### **Core LangChain:**
- OpenAI GPT integration (`langchain-openai`)
- Hugging Face models (`langchain-huggingface`) 
- Community tools (`langchain-community`)
- Prompt engineering (`langchain-core`)

### **Advanced Features:**
- **LangGraph** - Visual AI workflows
- **Vector Databases** - Chroma for document storage
- **Memory Systems** - Persistent conversation state
- **Streaming** - Real-time response generation

### **External Integrations:**
- **Wikipedia API** - Real-world information lookup
- **PDF Processing** - Document analysis and indexing
- **Custom Tools** - Mathematical functions and text processing

## 📁 Project Structure

```
langchain-mastery/
├── README.md                 # This comprehensive guide
├── .env-sample              # Environment template
├── .env                     # Your API keys (create from sample)
├── requirements.txt         # All Python dependencies
│
├── Scripts (14 total):
├── script_01.py            # Basic OpenAI integration
├── script_02.py            # Hugging Face models
├── script_03.py            # Prompt templates
├── script_04.py            # Chat templates
├── script_05.py            # Few-shot learning
├── script_06.py            # Sequential chains
├── script_07.py            # AI agents with tools
├── script_08.py            # Basic RAG system
├── script_09.py            # Advanced RAG system
├── script_10.py            # Custom tools
├── script_11.py            # Conversation memory
├── script_12.py            # LangGraph basics
├── script_13.py            # Smart research assistant
├── script_14.py            # Multi-tool AI assistant
│
├── Tutorials (14 total):
├── tutorial_01.md          # Basic OpenAI explained
├── tutorial_02.md          # Hugging Face explained
├── tutorial_03.md          # Prompt templates explained
├── tutorial_04.md          # Chat templates explained
├── tutorial_05.md          # Few-shot learning explained
├── tutorial_06.md          # Sequential chains explained
├── tutorial_07.md          # AI agents explained
├── tutorial_08.md          # Basic RAG explained
├── tutorial_09.md          # Advanced RAG explained
├── tutorial_10.md          # Custom tools explained
├── tutorial_11.md          # Conversation memory explained
├── tutorial_12.md          # LangGraph basics explained
├── tutorial_13.md          # Smart research assistant explained
├── tutorial_14.md          # Multi-tool AI assistant explained
│
├── Data & Resources:
├── pdfs/                   # PDF files for RAG systems
│   └── rag_vs_fine_tuning.pdf
├── chroma_db/              # Vector database (auto-created)
├── data/                   # Additional datasets
│   └── fifa_countries_audience.csv
│
└── Generated Diagrams:     # Auto-created workflow visualizations
    ├── chatbot_graph.png
    ├── wikipedia_chatbot_graph_nomemory.png
    ├── wikipedia_chatbot_graph_with_memory.png
    └── multi_tool_graph_with_memory.png
```

## 🔧 Troubleshooting Guide

### Common Setup Issues

**"No module named 'langchain'"**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**"OpenAI API key not found"**
```bash
# Check your .env file exists and contains:
OPENAI_API_KEY=sk-your-actual-key-here
# (without quotes, no extra spaces)
```

**"PDF not found" (scripts 08, 09)**
```bash
# Create directory and add PDF
mkdir -p pdfs
# Download or place your PDF at: pdfs/rag_vs_fine_tuning.pdf
```

### Script-Specific Notes

- **script_02.py**: Downloads ~1-2GB models on first run
- **script_07.py, script_13.py**: Require internet for Wikipedia API
- **script_08.py, script_09.py**: Create `chroma_db/` directory automatically
- **script_12.py, script_13.py, script_14.py**: Generate PNG workflow diagrams

### Advanced Features

**LangSmith Warnings**: These are harmless. Scripts automatically disable tracing.

**Memory Persistence**: Scripts 11-14 save conversation state automatically.

**Streaming Responses**: Scripts 12-14 show real-time AI responses.

## 🧠 What Each Tutorial Teaches

### Tutorial Highlights:

**Beginner Concepts:**
- ✅ Setting up API connections
- ✅ Understanding prompts vs templates
- ✅ Managing conversation context
- ✅ Building multi-step AI workflows

**Intermediate Concepts:**
- ✅ Creating intelligent AI agents
- ✅ Building document analysis systems
- ✅ Vector database operations
- ✅ Tool integration patterns

**Advanced Concepts:**
- ✅ Graph-based AI workflows
- ✅ Persistent memory management
- ✅ Real-time streaming interfaces
- ✅ Multi-tool system architecture

## 🌟 Key Features of Each Script

### Scripts 1-6: Foundation Building
- **Progressive complexity**: Each script builds on previous concepts
- **Clear examples**: Simple, focused demonstrations
- **Best practices**: Production-ready patterns from the start

### Scripts 7-9: Real-World Applications
- **Tool integration**: External APIs and services
- **Document processing**: PDF analysis and retrieval
- **Smart agents**: Decision-making and reasoning

### Scripts 10-14: Advanced Systems
- **Custom tools**: Create your own AI capabilities
- **Visual workflows**: See how your AI thinks
- **Production features**: Memory, streaming, error handling
- **Multi-tool systems**: Ultimate AI assistants

## 🎯 Next Steps After Completion

After working through all 14 scripts and tutorials, you'll be able to:

### **Build Production AI Systems:**
- Multi-agent workflows for complex tasks
- Document analysis pipelines for businesses
- Conversational AI with perfect memory
- Custom tools for specialized domains

### **Integrate with Real Applications:**
- Customer service chatbots
- Research and analysis tools
- Educational tutoring systems
- Business intelligence platforms

### **Explore Advanced Topics:**
- Multi-modal AI (text, images, audio)
- Large-scale RAG systems
- AI agent orchestration
- Custom model fine-tuning

## 🤝 Need Help?

### Each Tutorial Includes:
- **Line-by-line explanations** in plain English
- **Common issues and solutions** sections
- **Experiment suggestions** for further learning
- **Real-world application** examples

### Troubleshooting Steps:
1. Check the specific tutorial for your script
2. Verify your virtual environment is activated
3. Ensure your `.env` file is configured correctly
4. Check that required files (PDFs, etc.) are in place
5. Look at the console output for specific error messages

### Learning Resources:
- **Official LangChain docs**: [langchain.com](https://langchain.com)
- **LangGraph documentation**: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph)
- **OpenAI API docs**: [platform.openai.com/docs](https://platform.openai.com/docs)

## 💡 Why This Collection is Unique

Unlike other LangChain examples, this collection provides:

1. **Complete Learning Journey**: 14 progressive scripts covering the entire ecosystem
2. **Detailed Explanations**: Every line of code explained for beginners
3. **Production Patterns**: Real-world best practices, not just demos
4. **Visual Learning**: Automatic workflow diagrams for complex systems
5. **Memory & Persistence**: Advanced features often skipped in tutorials
6. **Multi-Tool Integration**: Shows how to combine different AI capabilities

**Perfect for:** Developers, AI enthusiasts, students, and professionals wanting to master LangChain from basics to advanced implementations.

## 🤝 Contributing

We welcome contributions to make this learning resource even better! Here's how you can help:

### **Ways to Contribute:**
- 🐛 **Report Issues**: Found a bug or error in tutorials? [Open an issue](https://github.com/datatweets/langchain-mastery/issues)
- 📝 **Improve Documentation**: Help make tutorials clearer for beginners
- 💡 **Suggest Features**: Ideas for new scripts or advanced examples
- 🔧 **Fix Bugs**: Submit pull requests with fixes
- ⭐ **Star the Repository**: Help others discover this resource

### **Getting Started:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and test thoroughly
4. Commit with clear messages (`git commit -m 'Add amazing feature'`)
5. Push to your branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📊 Repository Stats

- **🌟 Star this repository** if you find it helpful!
- **🍴 Fork it** to create your own version
- **👀 Watch** for updates and new tutorials
- **📢 Share** with fellow AI enthusiasts

## 📬 Connect & Support

- **Repository**: [github.com/datatweets/langchain-mastery](https://github.com/datatweets/langchain-mastery)
- **Issues**: [Report problems or request features](https://github.com/datatweets/langchain-mastery/issues)
- **Discussions**: [Join community discussions](https://github.com/datatweets/langchain-mastery/discussions)

---

**⭐ If this repository helped you learn LangChain, please give it a star!** ⭐

Happy learning! 🚀✨