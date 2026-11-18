# Multi-Agent Personal Assistant

## Overview
This repository contains a **Multi-Agent Personal Assistant** built with **FastAPI**, **LangGraph**, **Groq (`openai/gpt-oss-120b`)**, and **Composio**. Instead of relying on a single prompt to handle everything, the system uses a coordinated Supervisor–Worker approach where a central orchestrator routes tasks to specialized agents and chains their outputs together for end-to-end Google Workspace automation.

## Architecture
The workflow consists of interconnected components managed through a LangGraph state machine:

- **Input (FastAPI & Web UI):** Entry point where the user submits natural-language prompts or multi-step action items.
- **Orchestration Layer (`Supervisor Node`):** Analyzes user intent using structured Pydantic routing (`ExecutionPlan`) and decides the exact sequence of agents required.
- **Specialist Agents:**
  - **Assistant Agent:** Manages casual conversation, Q&A, and drafts clean, structured content.
  - **Google Docs Agent:** Creates and retrieves Google Documents via Composio and returns direct share/edit links.
  - **Google Calendar Agent:** Schedules meetings and checks events with IST timezone awareness and Google Meet integration.
  - **Gmail Agent:** Composes and saves clean email drafts or sends emails using context passed from earlier agents.
- **Synthesizer Node:** Combines outputs from all executed agents into a clean, human-readable confirmation response.

```
User Prompt ──▶ Supervisor Node (Plans Agent Chain)
                      │
                      ▼
       ┌────▶ Specialist Agent Node ◄────┐
       │      (Executes & passes         │
       │       accumulated_context)      │
       └────────── (Next step?) ─────────┘
                      │ No
                      ▼
               Synthesizer Node ──▶ Clean Plain-Text Output
```

## Features
- **Multi-Agent Chaining:** Handles multi-step requests in a single prompt (e.g., drafting notes, saving them to Google Docs, and preparing a Gmail draft with the document link).
- **Dynamic Task Routing:** Automatically selects only the required agents—using a single agent for casual chat and chaining multiple specialists for complex workflows.
- **Google Workspace Integration:** Seamless OAuth-based execution for Google Docs, Google Calendar, and Gmail through Composio.
- **Clean Plain-Text Formatting:** Automatically strips raw Markdown symbols (`##`, `**`, tables) before writing to Google Docs or Gmail drafts so documents and emails look ready to send.
- **Conversational Memory & Task History:** Maintains recent chat context and logs completed workspace tasks locally.

## Prerequisites
- Python 3.10 or higher
- A **Groq API Key** (for LLM inference)
- A **Composio API Key** with Google Docs, Google Calendar, and Gmail connected

## Getting Started
1. **Clone this repository:**
   git clone https://github.com/Gokulmalav/multi-agent-personal-assistant.git
   cd multi-agent-personal-assistant

2. **Install the required dependencies:**
   pip install -r requirements.txt

3. **Configure environment variables:**
   Create a `.env` file in the project root and add your API keys:
   GROQ_API_KEY=your_groq_api_key_here
   COMPOSIO_API_KEY=your_composio_api_key_here

4. **Run the flow and interact with your personal assistant**

## Usage
1. Open the web interface at `http://127.0.0.1:8000`.
2. Enter a natural-language prompt or multi-step task in the chat input.
3. Try sample prompts across different workflows:
   - **General Q&A:** *"Explain how the Supervisor-Worker pattern works in LangGraph."*
   - **Google Calendar:** *"Schedule a meeting tomorrow at 2 PM for Full Stack project viva preparation."*
   - **Gmail Draft:** *"Draft an email to my project guide requesting a review of my multi-agent assistant project."*
   - **Multi-Agent Chaining:** *"Write 5 key viva tips, save them in a Google Doc, and create an email draft with the document link."*
4. Watch the live agent execution pipeline in the UI and open the generated Google Doc, Calendar event, or Gmail draft directly from the links provided.

## Contributing
Contributions, suggestions, and improvements are welcome! Feel free to fork the repository and submit a Pull Request.

```
+-------------------------------------------------------------------------+
|                        PRESENTATION LAYER (Web UI)                      |
|      HTML5 / CSS3 / JavaScript  |  Chat Interface  |  Task History      |
+------------------------------------+------------------------------------+
                                     | HTTP POST (/api/chat) & GET (/api/tasks)
                                     v
+-------------------------------------------------------------------------+
|                     API & ORCHESTRATION LAYER (Backend)                 |
|  FastAPI Server (Uvicorn)  -->  LangGraph Stateful Workflow Engine      |
|                                                                         |
|   +-------------------+      +-------------------+      +------------+  |
|   |  Supervisor Node  | ---> |  Specialist Node  | ---> | Synthesizer|  |
|   | (Pydantic Plan)   |      | (Loop Execution)  |      |   Node     |  |
|   +-------------------+      +---------+---------+      +------------+  |
|                                        |                                |
|                    [Shared AgentState: accumulated_context]             |
+----------------------------------------+--------------------------------+
                                         | Managed OAuth & Tool Calling
                                         v
+-------------------------------------------------------------------------+
|                     EXTERNAL AI & WORKSPACE SERVICES                    |
|   Groq API (openai/gpt-oss-120b)   |   Composio SDK (Docs/Cal/Gmail)    |
+-------------------------------------------------------------------------+
```