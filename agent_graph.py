from typing import Annotated, Sequence, TypedDict
import operator

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

import sys
import os

# Import agent functions
from agent1_extractor import run_extractor
from agent2_fetcher import fetch_papers
from agent3_ingestor import run_ingestor
from agent5_batch_citer import run_batch_citer

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
    run_batch_citer(file_path, out_path)
    return f"Draft cited successfully. Output saved to {out_path}."

tools = [extract_citations_tool, fetch_papers_tool, ingest_papers_tool, batch_cite_tool]

# Initialize LLM
llm = ChatOllama(model="gemma4:latest", temperature=0)
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
