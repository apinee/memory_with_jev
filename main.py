"""
Main CLI Entrypoint: Memory-Graph Chatbot with Jev Memory-Persistence Gate
===========================================================================
Interactive terminal interface providing real-time visibility into the
Google Gemini conversational loop and TypeSafe Jev System One memory gating.

Commands:
- Type any message to chat.
- /stats or /graph   : Display summary of current graph nodes and categories.
- /visualize         : Render interactive HTML memory graph via visualize.py.
- /clear             : Clear the memory graph and reset.
- /exit or /quit     : Exit the application.
"""

import sys
import os
from dotenv import load_dotenv

# Safe UTF-8 output configuration for Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Load environment configuration
load_dotenv()

from agent import MemoryChatbotAgent
from memory_graph import MemoryGraph
from jev_gate import JevGate
from visualize import render_graph_html


def print_banner():
    banner = """
================================================================================
       MEMORY-GRAPH CHATBOT WITH TYPESAFE JEV DECISION GATE (SYSTEM ONE)
================================================================================
* Chat Model : Google Gemini (Conversational Engine)
* Memory Gate: TypeSafe Jev (Zero-generation, typed classification: Noul/Choice/Score)
* Graph Store: NetworkX -> local graph.json
--------------------------------------------------------------------------------
Commands: '/stats', '/visualize', '/clear', '/help', or 'exit' to quit.
================================================================================
"""
    print(banner)


def format_gate_badge(decision) -> str:
    """Format the Jev decision with status tags."""
    status_str = "PERSISTED TO GRAPH" if decision.should_persist else "DISCARDED (EPHEMERAL)"
    badge = (
        f"\n  +----------------------------------------------------------------------+\n"
        f"  | [JEV GATE DECISION] -> {status_str:<45} |\n"
        f"  | Should Persist (Noul) : {decision.probability:.2f} (Threshold: >= 0.65)           |\n"
        f"  | Category (Choice)     : {decision.category:<12}                                 |\n"
        f"  | Importance (Score)    : {decision.importance}/5 {'*' * decision.importance:<5}                               |\n"
        f"  | Engine Mode           : {'MOCK HEURISTIC' if decision.is_mock else 'TYPESAFE LIVE API':<45} |\n"
        f"  +----------------------------------------------------------------------+"
    )
    return badge


def main():
    print_banner()

    storage_path = "graph.json"
    memory_graph = MemoryGraph(storage_path=storage_path)
    jev_gate = JevGate()
    agent = MemoryChatbotAgent(
        memory_graph=memory_graph,
        jev_gate=jev_gate,
        graph_storage_path=storage_path
    )

    stats = memory_graph.get_stats()
    print(f"[*] Initialized. Existing long-term memory nodes: {stats['total_nodes']} (Edges: {stats['total_edges']})\n")

    while True:
        try:
            user_input = input("User > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting. Good bye!")
            break

        if not user_input:
            continue

        cmd_lower = user_input.lower()
        if cmd_lower in ["exit", "quit", "/exit", "/quit"]:
            print("\nExiting session. Long-term memories saved to graph.json.")
            break

        if cmd_lower in ["/stats", "/graph"]:
            st = memory_graph.get_stats()
            print(f"\n--- Current Memory Graph Status ---")
            print(f"Total Memory Nodes: {st['total_nodes']}")
            print(f"Total Graph Edges : {st['total_edges']}")
            print(f"Categories Breakdown: {st['categories']}")
            nodes = memory_graph.get_all_nodes()
            if nodes:
                print("\nStored Memories:")
                for n in nodes:
                    print(f"  * [{n['id']}] [{n['category'].upper()}] (Weight: {n['weight']}/5) {n['text'].splitlines()[0]}")
            print("------------------------------------\n")
            continue

        if cmd_lower in ["/visualize", "/vis"]:
            print("\nGenerating interactive HTML graph...")
            render_graph_html(json_path=storage_path, output_html="memory_graph.html", open_browser=False)
            print("[+] Open 'memory_graph.html' in your browser to view the interactive network!\n")
            continue

        if cmd_lower in ["/clear"]:
            memory_graph.clear()
            print("\n[!] Memory graph cleared and graph.json reset.\n")
            continue

        if cmd_lower in ["/help"]:
            print("\nAvailable commands:")
            print("  /stats     - Show current graph nodes and categories")
            print("  /visualize - Generate interactive HTML visualization (memory_graph.html)")
            print("  /clear     - Reset the memory graph")
            print("  exit       - Quit the chatbot\n")
            continue

        # Process conversation turn
        result = agent.process_turn(user_input)

        # Print retrieved memories if any
        if result.retrieved_memories:
            print(f"\n  [<- Retrieved {len(result.retrieved_memories)} Relevant Memory Nodes from Graph:]")
            for m in result.retrieved_memories:
                print(f"     - [{m['category'].upper()}] (wt={m['weight']}): {m['text'].splitlines()[0]}")

        # Print assistant reply
        print(f"\nAssistant > {result.assistant_reply}")

        # Print Jev gating breakdown
        print(format_gate_badge(result.gate_decision))

        if result.persisted_node_id:
            print(f"  --> Saved as Graph Node: {result.persisted_node_id} (graph.json updated)\n")
        else:
            print(f"  --> Discarded from long-term memory (not durable/informative)\n")


if __name__ == "__main__":
    main()
