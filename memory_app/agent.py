"""
Google ADK Agent Definition for TypeSafe Jev Memory Graph
==========================================================
Defines the `root_agent` discovered and rendered by the Google ADK Web Chat Interface.

Features:
- Powered by Google Gemini.
- Memory Retrieval tool & automatic context injection in `before_agent_callback`.
- Automatic System One gating via TypeSafe Jev in `after_agent_callback`.
- Persistent storage to `graph.json`.
"""

import os
import sys
import logging
from typing import Optional, Any
from dotenv import load_dotenv

# Ensure workspace root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

load_dotenv()

from google.adk.agents.llm_agent import Agent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from memory_graph import MemoryGraph
from jev_gate import JevGate

logger = logging.getLogger("GoogleADKMemoryApp")

# Singleton memory graph and Jev gate instances
STORAGE_PATH = os.path.join(parent_dir, "graph.json")
memory_graph = MemoryGraph(storage_path=STORAGE_PATH)
jev_gate = JevGate()


def query_memory_graph(query: str) -> str:
    """
    Search the long-term memory graph for relevant past facts, preferences, goals, or relationships.

    Args:
        query: Search keywords or question.

    Returns:
        Formatted summary of relevant memory nodes retrieved from the graph.
    """
    memories = memory_graph.query_relevant(query, top_k=5)
    if not memories:
        return "No relevant long-term memories found matching your query."

    output = ["### Retrieved Memories from Knowledge Graph:"]
    for m in memories:
        cat = m.get("category", "info").upper()
        wt = m.get("weight", 1)
        stars = "★" * wt + "☆" * (5 - wt)
        txt = m.get("text", "").replace("\n", " | ")
        output.append(f"- **[{cat}]** ({stars} {wt}/5) `{m.get('id')}`: {txt}")
    return "\n".join(output)


def view_memory_graph_stats() -> str:
    """
    Get detailed breakdown of all active nodes and categories currently stored in the memory graph state.

    Returns:
        Structured table and statistics of all stored knowledge in graph.json.
    """
    st = memory_graph.get_stats()
    nodes = memory_graph.get_all_nodes()
    
    lines = [
        "### 🧠 Long-Term Memory Graph State",
        f"- **Total Stored Nodes:** {st['total_nodes']}",
        f"- **Total Relational Edges:** {st['total_edges']}",
        f"- **Category Breakdown:** {st['categories']}",
        "\n| Node ID | Category | Importance | Content Snippet |",
        "| :--- | :--- | :--- | :--- |"
    ]
    
    if not nodes:
        lines.append("| _(Empty)_ | _(None)_ | _(0/5)_ | _No memories stored yet. Chat to build graph._ |")
    else:
        for n in nodes:
            node_id = n.get("id", "node")
            cat = n.get("category", "info").upper()
            wt = n.get("weight", 1)
            first_line = n.get("text", "").splitlines()[0]
            if len(first_line) > 55:
                first_line = first_line[:52] + "..."
            first_line = first_line.replace("|", "/")
            lines.append(f"| `{node_id}` | **{cat}** | {'★' * wt} ({wt}/5) | {first_line} |")

    return "\n".join(lines)


async def before_agent_callback(ctx: CallbackContext) -> Optional[types.Content]:
    """
    Before Agent Hook:
    Extracts the user message, queries MemoryGraph for top relevant nodes,
    and updates session state with the current memory state.
    """
    try:
        user_msg = ""
        if hasattr(ctx, "user_content") and ctx.user_content:
            for part in ctx.user_content.parts:
                if getattr(part, "text", None):
                    user_msg += part.text

        # Synchronize live memory graph snapshot into ADK session state
        st = memory_graph.get_stats()
        all_nodes = memory_graph.get_all_nodes()
        ctx.state["memory_graph_summary"] = {
            "total_nodes": st["total_nodes"],
            "total_edges": st["total_edges"],
            "categories": st["categories"]
        }
        ctx.state["active_memory_nodes"] = [
            {
                "id": n.get("id"),
                "category": n.get("category"),
                "importance": n.get("weight"),
                "snippet": n.get("text", "").splitlines()[0],
                "timestamp": n.get("timestamp")
            }
            for n in all_nodes
        ]

        if user_msg:
            memories = memory_graph.query_relevant(user_msg, top_k=3)
            if memories:
                snippets = []
                for m in memories:
                    snippets.append(f"[{m['category'].upper()} | Wt:{m['weight']}]: {m['text'].splitlines()[0]}")
                context_str = " | ".join(snippets)
                ctx.state["retrieved_context_for_current_turn"] = context_str
                logger.info(f"[ADK Before Callback] Injected memory context: {context_str}")
            else:
                ctx.state["retrieved_context_for_current_turn"] = None
    except Exception as e:
        logger.warning(f"[before_agent_callback] Error: {e}")

    return None


async def after_agent_callback(ctx: CallbackContext) -> Optional[types.Content]:
    """
    After Agent Hook:
    Captures completed turn, sends to TypeSafe Jev System One, persists if approved,
    and publishes full decision telemetry into session state.
    """
    try:
        user_text = ""
        agent_text = ""

        # Extract user and model text from session events
        if ctx.session and ctx.session.events:
            for event in reversed(ctx.session.events):
                if hasattr(event, "content") and event.content:
                    role = getattr(event.content, "role", "")
                    text = ""
                    for p in event.content.parts:
                        if getattr(p, "text", None):
                            text += p.text

                    if role == "model" and not agent_text:
                        agent_text = text
                    elif role == "user" and not user_text:
                        user_text = text

                    if user_text and agent_text:
                        break

        if user_text and agent_text:
            # Package turn into string (no extra LLM call)
            packaged_turn = f"User: {user_text}\nAssistant: {agent_text}"

            # Evaluate with TypeSafe Jev Gate
            decision = jev_gate.evaluate_turn(packaged_turn)
            logger.info(f"[ADK After Callback] Jev Decision: {decision.summary()}")

            persisted_node_id = None
            if decision.should_persist:
                turn_id = len(memory_graph.get_all_nodes()) + 1
                persisted_node_id = memory_graph.add_node(
                    turn_id=turn_id,
                    text=packaged_turn,
                    category=decision.category,
                    weight=decision.importance
                )
                logger.info(f"[ADK After Callback] Memory persisted as node: {persisted_node_id}")

            # Update live state in ADK UI state inspector
            st = memory_graph.get_stats()
            all_nodes = memory_graph.get_all_nodes()
            ctx.state["last_turn_jev_gating"] = {
                "decision": "PERSISTED" if decision.should_persist else "DISCARDED",
                "should_persist_probability": round(decision.probability, 3),
                "category": decision.category,
                "importance_score": f"{decision.importance}/5",
                "persisted_node_id": persisted_node_id,
                "is_mock": decision.is_mock
            }
            ctx.state["memory_graph_summary"] = {
                "total_nodes": st["total_nodes"],
                "total_edges": st["total_edges"],
                "categories": st["categories"]
            }
            ctx.state["active_memory_nodes"] = [
                {
                    "id": n.get("id"),
                    "category": n.get("category"),
                    "importance": n.get("weight"),
                    "snippet": n.get("text", "").splitlines()[0],
                    "timestamp": n.get("timestamp")
                }
                for n in all_nodes
            ]
    except Exception as e:
        logger.warning(f"[after_agent_callback] Error in Jev gating: {e}")

    return None


# Construct root_agent discovered by ADK Web Server
root_agent = Agent(
    name="memory_graph_agent",
    model="gemini-flash-lite-latest",
    description="Conversational AI agent equipped with a TypeSafe Jev gated long-term memory graph.",
    instruction=(
        "You are a friendly, helpful, and attentive AI assistant equipped with a dynamic long-term memory graph.\n"
        "- When relevant user facts, preferences, goals, or relationships are retrieved from the memory graph, "
        "incorporate them naturally into your responses.\n"
        "- When the user asks what is stored in memory, what you remember, or asks to see the memory graph/state, "
        "always use the `view_memory_graph_stats` tool to display the complete memory state table.\n"
        "- You can use `query_memory_graph` to search for specific memories anytime.\n"
        "- Every turn you complete is evaluated automatically by TypeSafe Jev (a System One decision model) "
        "to decide whether durable facts should be written into your permanent knowledge graph."
    ),
    tools=[query_memory_graph, view_memory_graph_stats],
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback
)
