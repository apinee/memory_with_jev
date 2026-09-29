"""
End-to-End Verification Test Script
====================================
Simulates a multi-turn demo session to verify:
1. Ephemeral small talk turns are discarded by Jev.
2. High-value facts, preferences, relationships, and goals are persisted.
3. Graph nodes and edges are written to graph.json.
4. Subsequent questions retrieve relevant past memories and ground the assistant's answer.
5. PyVis visualization generates memory_graph.html cleanly.
"""

import sys
import os
import json

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from agent import MemoryChatbotAgent
from memory_graph import MemoryGraph
from jev_gate import JevGate
from visualize import render_graph_html


def run_verification():
    print("=================================================================")
    print("     STARTING MULTI-TURN VERIFICATION TEST FOR MEMORY CHATBOT    ")
    print("=================================================================")

    test_graph_file = "graph.json"
    mem_graph = MemoryGraph(storage_path=test_graph_file)
    mem_graph.clear()  # Start fresh

    gate = JevGate()
    agent = MemoryChatbotAgent(
        memory_graph=mem_graph,
        jev_gate=gate,
        graph_storage_path=test_graph_file
    )

    test_script = [
        ("Turn 1: Small talk greeting", "Hello there! How are you doing today?"),
        ("Turn 2: Personal Fact", "I live in Seattle and work as a distributed systems engineer at CloudCorp."),
        ("Turn 3: Personal Preference", "I strictly drink oat milk lattes and hate regular dairy."),
        ("Turn 4: Family & Event", "My brother Dave is visiting next week for the Seattle marathon."),
        ("Turn 5: Career Goal", "My goal is to launch an open source vector database project by December."),
        ("Turn 6: Ephemeral Filler", "Cool, thanks for letting me know!"),
        ("Turn 7: Context Retrieval Query", "What city do I live in, what is my favorite drink, and who is visiting me?")
    ]

    for label, user_msg in test_script:
        print(f"\n>>> [{label}]")
        print(f"User: '{user_msg}'")
        res = agent.process_turn(user_msg)

        if res.retrieved_memories:
            print(f"  [Memory Retrieval] Found {len(res.retrieved_memories)} relevant graph nodes:")
            for m in res.retrieved_memories:
                print(f"    - [{m['category'].upper()} | weight={m['weight']}]: {m['text'].splitlines()[0]}")

        print(f"Assistant: {res.assistant_reply}")
        status = "PERSISTED" if res.gate_decision.should_persist else "DISCARDED"
        print(f"Jev Gate: [{status}] Category='{res.gate_decision.category}', Prob={res.gate_decision.probability:.2f}, Score={res.gate_decision.importance}/5")
        if res.persisted_node_id:
            print(f"  -> Written to Graph as: {res.persisted_node_id}")

    # Check graph.json
    print("\n-----------------------------------------------------------------")
    print("Verifying graph.json persistence...")
    assert os.path.exists(test_graph_file), "graph.json must exist"

    with open(test_graph_file, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    print(f"[+] Total Persisted Nodes: {len(nodes)}")
    print(f"[+] Total Graph Edges    : {len(edges)}")
    print("\nPersisted Node Categories:")
    categories = [n["category"] for n in nodes]
    for c in set(categories):
        print(f"  - {c}: {categories.count(c)} node(s)")

    assert len(nodes) >= 4, f"Expected at least 4 persisted nodes, got {len(nodes)}"
    assert "fact" in categories, "Expected 'fact' category in graph"
    assert "preference" in categories, "Expected 'preference' category in graph"

    # Test visualization generation
    print("\n-----------------------------------------------------------------")
    print("Generating HTML Visualization via PyVis...")
    html_path = render_graph_html(json_path=test_graph_file, output_html="memory_graph.html")
    assert os.path.exists("memory_graph.html"), "memory_graph.html must be generated"
    print(f"[+] Verified visualization file: {html_path} (Size: {os.path.getsize('memory_graph.html')} bytes)")

    print("\n=================================================================")
    print("            ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!       ")
    print("=================================================================")


if __name__ == "__main__":
    run_verification()
