from typing import Annotated, Sequence, TypedDict
import operator

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

from config import LLM_MODEL

# Import agent functions
from agent1_extractor import run_extractor
from agent2_fetcher import fetch_papers
from agent3_ingestor import run_ingestor
from agent5_batch_citer import run_batch_citer
from agent6_manual_ingestor import ingest_manual_pdf
from agent7_research_chat import ResearchChat

@tool
def extract_citations_tool():
    """Extracts citations from all new PDFs in the raw directory. 
    Use this first when a new PDF is dropped."""
    run_extractor()
    return "Citations extracted successfully."

@tool
def fetch_papers_tool():
    """Fetches full papers (PDFs) based on the extracted citations.
    Use this after extracting citations."""
    fetch_papers()
    return "Papers fetched successfully."

@tool
def ingest_papers_tool(workers: int = 4):
    """Ingests the fetched papers into the vector database and builds the BM25 index.
    Use this after fetching papers."""
    run_ingestor(workers=workers)
    return "Papers ingested successfully."

@tool
def batch_cite_tool(file_path: str):
    """Adds citations to a draft text file and saves the result.
    Use this when a new draft text file needs to be cited."""
    out_path = file_path.replace(".txt", "_cited.txt")
    try:
        run_batch_citer(file_path, out_path)
    except RuntimeError as e:
        return f"Tool failed: {e} (Have papers been ingested yet?)"
    return f"Draft cited successfully. Output saved to {out_path}."

@tool
def manual_ingest_tool(pdf_path: str, citation_string: str = ""):
    """Ingests a single manually placed PDF from the pulled_pdfs/ directory into
    the ChromaDB vector database and rebuilds the BM25 index.
    Use this when a user manually drops a PDF into the pulled_pdfs/ directory.
    Optionally accepts a citation_string label; defaults to the PDF filename."""
    ingest_manual_pdf(pdf_path, citation_string=citation_string or None)
    return f"Manual PDF '{pdf_path}' ingested successfully."

# Lazy-init singleton so we don't load the DB at import time
_research_agent = None

@tool
def research_chat_tool(question: str):
    """Answers a research question using the ingested paper database.
    Use this when a user asks a conceptual, exploratory, or brainstorming question
    about their research area. The tool searches the paper database and returns
    a grounded answer with source citations."""
    global _research_agent
    try:
        if _research_agent is None:
            _research_agent = ResearchChat(top_k=5)
        answer = _research_agent.chat(question)
        return answer
    except RuntimeError as e:
        return f"Tool failed: {e} (Have papers been ingested yet?)"

tools = [extract_citations_tool, fetch_papers_tool, ingest_papers_tool, batch_cite_tool, manual_ingest_tool, research_chat_tool]

# Initialize LLM
llm = ChatOllama(model=LLM_MODEL, temperature=0)
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def agent_node(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Build Graph
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()

def process_event(message: str):
    print(f"\n--- LangGraph Agent Triggered ---")
    print(f"Message: {message}")
    state = {"messages": [HumanMessage(content=message)]}
    for event in graph.stream(state, stream_mode="values"):
        last_msg = event["messages"][-1]
        last_msg.pretty_print()
    print("--- LangGraph Agent Finished ---\n")

if __name__ == "__main__":
    process_event("A new PDF was dropped in the raw directory. Please process it.")
