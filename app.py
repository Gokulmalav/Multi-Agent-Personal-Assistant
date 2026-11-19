import os
import re
import json
from datetime import datetime
from typing import List, Literal
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from composio import Composio
from composio_langchain import LangchainProvider
from langchain_groq import ChatGroq

# 1. API KEYS SETUP

from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["COMPOSIO_API_KEY"] = os.getenv("COMPOSIO_API_KEY", "")

# Initialize Groq LLM
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_retries=1, request_timeout=20)

# Initialize Composio SDK for Google Workspace
composio = Composio(provider=LangchainProvider())
USER_ID = "pg-test-3e0e5ce0-263e-4b77-abce-7575cf8b160f"

gdocs_tools = composio.tools.get(
    user_id=USER_ID,
    tools=["GOOGLEDOCS_CREATE_DOCUMENT", "GOOGLEDOCS_GET_DOCUMENT_BY_ID"]
)
gcal_tools = composio.tools.get(
    user_id=USER_ID,
    tools=["GOOGLECALENDAR_CREATE_EVENT", "GOOGLECALENDAR_FIND_EVENT"]
)
gmail_tools = composio.tools.get(
    user_id=USER_ID,
    tools=["GMAIL_CREATE_EMAIL_DRAFT", "GMAIL_SEND_EMAIL"]
)

# 2. LOCAL TASK HISTORY STORAGE

TASKS_FILE = "tasks.json"
CHAT_HISTORY = []

def load_tasks():
    if not os.path.exists(TASKS_FILE):
        return []
    try:
        with open(TASKS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_task_log(task_summary: str):
    tasks = load_tasks()
    tasks.insert(0, task_summary)
    with open(TASKS_FILE, "w") as f:
        json.dump(tasks[:15], f, indent=2)


# 3. SUPERVISOR ROUTING SCHEMA (MULTI-AGENT CHAINING)
class ExecutionPlan(BaseModel):
    agents_chain: List[Literal["assistant", "gdocs", "gcalendar", "gmail"]] = Field(
        description="Ordered list of specialist agents needed to complete the user request. E.g., ['assistant', 'gdocs', 'gmail'] if user wants content written, saved in a doc, and drafted in email."
    )
    reasoning: str = Field(description="Brief explanation of why these agents were chosen.")

class AgentState(MessagesState):
    plan: List[str]
    current_step: int
    accumulated_context: str
    step_logs: List[dict]

# 4. COMPOSIO TOOL EXECUTION HELPER & TEXT CLEANER
def clean_plain_text(text: str) -> str:
    """Removes Markdown symbols (##, **, |, `) so Docs, Emails, and UI look clean."""
    if not text:
        return ""
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = text.replace('`', '')
    text = re.sub(r'^\|?[\s\-:|]+\|[\s\-:|]*$', '', text, flags=re.MULTILINE)
    lines = []
    for line in text.splitlines():
        if line.strip().startswith('|') and line.strip().endswith('|'):
            cells = [c.strip() for c in line.strip('|').split('|') if c.strip()]
            lines.append(" — ".join(cells))
        else:
            lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def run_composio_tools(ai_msg):
    results = []
    for tc in ai_msg.tool_calls:
        args = tc["args"]
        for key in ["text", "content", "body"]:
            if key in args and isinstance(args[key], str):
                args[key] = clean_plain_text(args[key])

        res = composio.tools.execute(
            slug=tc["name"],
            arguments=args,
            user_id=USER_ID,
            dangerously_skip_version_check=True
        )
        if isinstance(res, dict):
            compact = {"successful": res.get("successful", True), "data": res.get("data", res)}
            data_dict = compact["data"] if isinstance(compact["data"], dict) else {}
            doc_id = data_dict.get("documentId") or data_dict.get("id")
            if tc["name"].startswith("GOOGLEDOCS") and doc_id:
                compact["document_url"] = f"https://docs.google.com/document/d/{doc_id}/edit"
            results.append(json.dumps(compact)[:1200])
        else:
            results.append(str(res)[:1200])
    return results

# 5. AGENT NODES

def supervisor_node(state: AgentState):
    """Analyzes user prompt and builds an ordered multi-agent execution plan."""
    current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    valid_agents = {"assistant", "gdocs", "gcalendar", "gmail"}
    plan = ["assistant"]

    sys_msg = SystemMessage(content=f"""You are the Supervisor Orchestrator of a Multi-Agent Personal Assistant.
Current Desktop Date & Time: {current_time}
Available Specialist Agents:
- 'assistant': General AI assistant for normal casual chat, greetings, answering questions, coding, or drafting content.
- 'gdocs': Google Docs Agent to create or read real Google Docs.
- 'gcalendar': Google Calendar Agent to schedule meetings/events or check schedule.
- 'gmail': Gmail Agent to draft or send emails.

Rules:
- For casual chat, greetings ('hi', 'hello', 'kaise ho'), or general Q&A, return ONLY ['assistant'].
- Always include at least one agent.
- Output ONLY valid JSON in this exact format: {{"agents_chain": ["assistant"]}}""")

    try:
        response = llm.invoke([sys_msg, state["messages"][0]])
        raw_text = response.content.strip()

        json_match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            extracted = [a for a in parsed.get("agents_chain", []) if a in valid_agents]
            if extracted:
                plan = extracted
        else:
            found = re.findall(r"\b(assistant|gdocs|gcalendar|gmail)\b", raw_text.lower())
            if found:
                plan = list(dict.fromkeys(found))
    except Exception as e:
        print(f"Supervisor fallback triggered: {e}")
        plan = ["assistant"]

    return {
        "plan": plan,
        "current_step": 0,
        "accumulated_context": "",
        "step_logs": []
    }

def execute_specialist_node(state: AgentState):
    """Executes the current specialist agent in the planned chain."""
    idx = state["current_step"]
    agent_name = state["plan"][idx]
    user_query = state["messages"][0].content
    prev_context = state["accumulated_context"]
    current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p (IST)")

    if agent_name == "assistant":
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in CHAT_HISTORY[-6:]])
        prompt = f"""You are a friendly, smart, and helpful Personal AI Assistant. Talk naturally like a helpful friend/collaborator (match the user's language—English or Hinglish).
Current Time: {current_time}
Recent Conversation History:
{history_text}

User Message: {user_query}
Previous Agent Context (if any): {prev_context}

IMPORTANT FORMATTING RULE:
- Write in clean PLAIN TEXT (NO Markdown tables '|', NO '##', NO '**' asterisks, NO backticks).
- If generating notes, summaries, or questions for a document, output ONLY the clean ready-to-use content with numbered headings and bullet points (•)."""
        res = clean_plain_text(llm.invoke(prompt).content)
        log_msg = "Conversational AI response"

    elif agent_name == "gdocs":
        llm_with_docs = llm.bind_tools(gdocs_tools)
        prompt = f"""You are the Google Docs Agent. User Request: {user_query}
Content/Context from previous agents: {prev_context}
Use your Google Docs tools to create or read the requested document. Put clean plain text (no **, ##, or |) inside the document, and always return the Document Title and Document URL."""
        ai_msg = llm_with_docs.invoke(prompt)
        if ai_msg.tool_calls:
            tool_out = run_composio_tools(ai_msg)
            final_res = f"Google Docs Operation Result: {tool_out[0]}"
        else:
            final_res = clean_plain_text(ai_msg.content)
        res = final_res
        log_msg = "Executed Google Docs operation"

    elif agent_name == "gcalendar":
        llm_with_cal = llm.bind_tools(gcal_tools)
        prompt = f"You are the Google Calendar Agent. Current Desktop Date & Time is: {current_time}.\nUser Request: {user_query}\nContext: {prev_context}\nSchedule or check the event accurately relative to current time."
        ai_msg = llm_with_cal.invoke(prompt)
        if ai_msg.tool_calls:
            tool_out = run_composio_tools(ai_msg)
            final_res = f"Google Calendar Operation Result: {tool_out[0]}"
        else:
            final_res = clean_plain_text(ai_msg.content)
        res = final_res
        log_msg = "Updated Google Calendar schedule"

    elif agent_name == "gmail":
        llm_with_gmail = llm.bind_tools(gmail_tools)
        prompt = f"""You are the Gmail Agent for Gokul Malav.
User Request: {user_query}
Content/Links from previous agents to include: {prev_context}
Create a clean, polite email draft or send the email as requested.
RULES:
- If no recipient email is given in the prompt, use recipient_email="recipient@example.com" so the draft is saved in Gmail.
- Write email body in clean PLAIN TEXT (NO **, NO ##, NO Markdown tables, NO backticks).
- Do NOT invent fake attachments (like Demo.mp4 or tests.zip).
- Sign off as Gokul Malav."""
        ai_msg = llm_with_gmail.invoke(prompt)
        if ai_msg.tool_calls:
            tool_out = run_composio_tools(ai_msg)
            final_res = f"Gmail Operation Result: {tool_out[0]}"
        else:
            final_res = clean_plain_text(ai_msg.content)
        res = final_res
        log_msg = "Processed Gmail draft/message"

    updated_context = f"{prev_context}\n\n--- Output from {agent_name.upper()} Agent ---\n{res}"
    updated_logs = state["step_logs"] + [{"agent": agent_name, "status": log_msg, "output": res}]

    return {
        "current_step": idx + 1,
        "accumulated_context": updated_context,
        "step_logs": updated_logs
    }

def final_synthesizer_node(state: AgentState):
    """Returns direct conversational reply for simple chat, or full summary for multi-agent tasks."""
    user_query = state["messages"][0].content
    
    # Agar sirf normal chat thi (['assistant']), toh seedha natural reply return karo
    if state["plan"] == ["assistant"]:
        direct_reply = state["step_logs"][0]["output"]
        CHAT_HISTORY.append({"role": "User", "content": user_query})
        CHAT_HISTORY.append({"role": "Assistant", "content": direct_reply})
        return {"messages": [HumanMessage(content=direct_reply)]}

    # Agar Workspace tools (Docs/Calendar/Gmail) chale hain, toh final summary banao
    all_outputs = state["accumulated_context"]
    summary_prompt = f"""User asked: {user_query}
All Agent Execution Results:
{all_outputs}

Write a clean, polite, well-formatted final summary of everything accomplished in PLAIN TEXT (NO Markdown tables '|', NO '##', NO '**' asterisks, NO backticks). Include any Google Doc links, Calendar event times, or Email confirmation details clearly."""
    final_msg = llm.invoke(summary_prompt)
    clean_final_text = clean_plain_text(final_msg.content)
    CHAT_HISTORY.append({"role": "User", "content": user_query})
    CHAT_HISTORY.append({"role": "Assistant", "content": clean_final_text})
    save_task_log(user_query[:55] + ("..." if len(user_query) > 55 else ""))
    return {"messages": [HumanMessage(content=clean_final_text)]}

def should_continue(state: AgentState):
    if state["current_step"] < len(state["plan"]):
        return "execute_specialist"
    return "synthesizer"

# 6. BUILD LANGGRAPH WORKFLOW

workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("execute_specialist", execute_specialist_node)
workflow.add_node("synthesizer", final_synthesizer_node)

workflow.add_edge(START, "supervisor")
workflow.add_edge("supervisor", "execute_specialist")
workflow.add_conditional_edges("execute_specialist", should_continue)
workflow.add_edge("synthesizer", END)

agent_app = workflow.compile()


# 7. FASTAPI SERVER & ENDPOINTS
app = FastAPI(title="Multi-Agent Personal Assistant")
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

@app.get("/api/tasks")
def get_tasks():
    return {"tasks": load_tasks()}

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    try:
        result = agent_app.invoke({
            "messages": [HumanMessage(content=req.message)],
            "plan": [],
            "current_step": 0,
            "accumulated_context": "",
            "step_logs": []
        })
        return {
            "status": "success",
            "plan": result["plan"],
            "steps": result["step_logs"],
            "reply": result["messages"][-1].content
        }
    except Exception as e:
        return {"status": "error", "reply": f"⚠️ Execution Error: {str(e)}", "plan": [], "steps": []}