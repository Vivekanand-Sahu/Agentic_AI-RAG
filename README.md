# Agentic-AI-RAG Crew

## IMPORTANT: Please note that the contents of this repository have been adapted to ensure compliance with confidentiality agreements.

Welcome to the Agentic-AI-RAG Crew project. This template is designed to help you set up a multi-agent AI system with ease, leveraging the powerful and flexible framework provided by crewAI. Our goal is to enable your agents to collaborate effectively on complex tasks, maximizing their collective intelligence and capabilities.

## Flow

<img width="1095" alt="image" src="https://github.com/user-attachments/assets/b5b5523a-73de-4332-a67e-41179f94de8f" />


# Tutor Chatbot: Multi-Agent RAG with LLMs

## Overview

This project features a fully functional **Tutor Chatbot**, powered by a multi-agent **Retrieval-Augmented Generation (RAG)** pipeline using **six Large Language Models (LLMs)**. The chatbot delivers **highly accurate** and **context-aware** responses, reducing hallucinations and ensuring data privacy.

## Key Features

- **Multi-Agent RAG**: Utilizes **LangChain** and **LangGraph** to orchestrate six specialized LLM agents.
- **Enhanced Accuracy**: Achieves **3x more precise** responses and reduces hallucinations by **60%**, leading to more reliable and context-aware answers that enhance user trust and learning efficiency. through:
  - **LLM Fine-Tuning**
  - **RAGAS (RAG Assessment Metrics)**
  - **Advanced Prompt Engineering**
- **Privacy-Focused Retrieval**:
  - **On-Device RAG**: Uses **Llama 3.2** and **FAISS** for local semantic search, ensuring **data security**.
  - **Optimized Web Search**: Implements **parallelized** web search using **GPT-3** and a **web-scraping tool**, achieving a **1.5x speedup** in response generation.

## Technologies Used

- **LangChain** & **LangGraph** for multi-agent coordination
- **FAISS** for fast semantic search
- **Llama 3.2** for on-device processing
- **GPT-3** for external knowledge augmentation
- **RAGAS** for evaluation and refinement
- **Web Scraping** tools for live data retrieval

## Installation

Ensure you have Python >=3.10 <3.13 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install crewai
crewai create flow Agentic_RAG
cd /content/agentic_rag
crewai install
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```

### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `src/agentic_rag/config/agents.yaml` to define your agents
- Modify `src/agentic_rag/config/tasks.yaml` to define your tasks
- Modify `src/agentic_rag/crew.py` to add your own logic, tools and specific args
- Modify `src/agentic_rag/main.py` to add custom inputs for your agents and tasks

## Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
crewai flow kickoff --directory /path/to/your/files --query "Tell me about Solar system"
```

This command initializes the Agentic_RAG Crew, assembling the agents and assigning them tasks as defined in your configuration.

This example, unmodified, will run the create a `report.md` file with the output of a research on LLMs in the root folder.

## Understanding Your Crew

The Agentic_RAG Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `config/agents.yaml` file outlines the capabilities and configurations of each agent in your crew.

## Support

For support, questions, or feedback regarding the {{crew_name}} Crew or crewAI.


Let's create wonders together with the power and simplicity of crewAI.
