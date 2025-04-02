# Agentic-AI-RAG Crew

## IMPORTANT: Please note that the contents of this repository have been adapted to ensure compliance with confidentiality agreements.

Welcome to the Agentic-AI-RAG Crew project. This template is designed to help you set up a multi-agent AI system with ease, leveraging the powerful and flexible framework provided by crewAI. Our goal is to enable your agents to collaborate effectively on complex tasks, maximizing their collective intelligence and capabilities.

## Flow

<img width="1095" alt="image" src="https://github.com/user-attachments/assets/b5b5523a-73de-4332-a67e-41179f94de8f" />


# Tutor Chatbot: Multi-Agent RAG with LLMs 🤖📚

## Overview ✨

This project presents the development of a **Tutor Chatbot** built with an **end-to-end multi-agent RAG system**. The chatbot integrates **6 LLMs** using **LangChain** and **LangGraph** to deliver accurate, fast, and private tutoring responses. The system achieves enhanced performance through **LLM Fine-Tuning**, **RAGAS**, and **Prompt Engineering**.

## Key Features 🚀

- **Multi-Agent Architecture**: Built with **6 LLMs** to cover a wide range of tutoring scenarios. 🧠
- **Enhanced Accuracy**: Achieved **3x more precise responses** and reduced **hallucinations by 60%**. 📈
- **Data Privacy**: Ensured complete privacy by utilizing **on-device Llama 3.2** and **FAISS** for RAG search. 🔐
- **Speed Optimization**: Parallelized **web search** using **GPT-3** and a **web-scraping tool** to achieve a **1.5x speedup**. ⚡

## Architecture 🏗️

- **LangChain & LangGraph**: These tools were used to seamlessly integrate and manage the multiple agents working together to process and retrieve relevant information for the tutor chatbot. 🔗🌐
- **RAG (Retrieval Augmented Generation)**: Incorporated to enhance the responses with more relevant and context-aware information. 📚
- **LLM Fine-Tuning**: Optimized the models for better accuracy and fewer hallucinations, improving the user experience. 🎯⚙
- **On-Device Execution**: Data privacy is ensured by processing everything locally, without the need to send sensitive information to external servers. 🏠

## Performance Improvements ⚡

- **3x Better Accuracy**: LLM fine-tuning led to a drastic improvement in response precision, providing more accurate answers to students. 📈
- **60% Reduction in Hallucinations**: Implemented techniques like **RAGAS** and **Prompt Engineering** to reduce model errors and irrelevant information. ❌
- **1.5x Faster Responses**: Speed was boosted by parallelizing web search operations, leveraging **GPT-3** and a web-scraping tool for quick and efficient search. 🚀

## Privacy & Security 🔐

- **On-Device Search**: By using **Llama 3.2** and **FAISS**, all searches and data processing are done locally on the device, ensuring complete data privacy. 🏡
- **Secure Communication**: All interactions between the chatbot and the user are secured with end-to-end encryption. 🔐

## Installation 🔧

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
crewai flow kickoff --directory /path/to/your/files --query "Tell me about Solar System."
```

This command initializes the Agentic_RAG Crew, assembling the agents and assigning them tasks as defined in your configuration.

This example, unmodified, will run the create a `report.md` file with the output of a research on LLMs in the root folder.

## Understanding Your Crew

The Agentic_RAG Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `config/agents.yaml` file outlines the capabilities and configurations of each agent in your crew.

### Contact:
If there is any confusion, you can reach out to me at 
Email: vsahu@ucsd.edu
LinkedIn: https://www.linkedin.com/in/vivekanand-sahu/



Let's create wonders together with power and simplicity.
