"""LangGraph runtime — minimal, provider-agnostic scaffold.

This serves as the live endpoint for langchain.axonbi.com. It ships with a
trivial LangGraph graph (no LLM call, no API key needed) purely to prove the
runtime is wired correctly. Replace `_graph` and add routes when you build the
real application.
"""
import os
import platform
from importlib.metadata import version, PackageNotFoundError
from typing import TypedDict

from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

app = FastAPI(title="LangGraph Runtime", version="0.1.0")


def _v(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "not-installed"


# --- Minimal demo graph: echoes input. No LLM / no API key required. ---
class EchoState(TypedDict):
    message: str
    reply: str


def _echo_node(state: EchoState) -> EchoState:
    return {"message": state["message"], "reply": f"echo: {state['message']}"}


_builder = StateGraph(EchoState)
_builder.add_node("echo", _echo_node)
_builder.add_edge(START, "echo")
_builder.add_edge("echo", END)
_graph = _builder.compile()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "service": "langgraph-runtime",
        "python": platform.python_version(),
        "versions": {
            "langchain": _v("langchain"),
            "langchain-core": _v("langchain-core"),
            "langgraph": _v("langgraph"),
            "langchain-anthropic": _v("langchain-anthropic"),
            "langchain-openai": _v("langchain-openai"),
            "fastapi": _v("fastapi"),
        },
        "llm_keys_configured": {
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY")),
        },
        "graph_deployed": False,
        "note": "Runtime only. Replace the demo graph with your application.",
    }


class EchoIn(BaseModel):
    message: str


@app.post("/echo")
def echo(body: EchoIn):
    return _graph.invoke({"message": body.message, "reply": ""})
