"""
Memory Graph Store Module
==========================
Wraps NetworkX to manage an in-memory knowledge graph of persisted conversation
memories, serializable to/from a local JSON file.

Key Features:
- Node creation with category, importance weight, timestamp, and text.
- Relational edge generation (sequential links and shared category links).
- Fast keyword/substring query retrieval to feed relevant past memory into agent context.
- Robust JSON persistence between runs.
"""

import os
import json
import re
import datetime
from typing import List, Dict, Any, Optional
import networkx as nx


class MemoryGraph:
    """
    Long-term memory graph powered by NetworkX and persisted to a local JSON file.
    """

    def __init__(self, storage_path: str = "graph.json"):
        """
        Initialize the memory graph.

        Args:
            storage_path: Path to the JSON persistence file.
        """
        self.storage_path = storage_path
        self.graph = nx.DiGraph()
        self._last_node_id: Optional[str] = None
        self.load_from_json()

    def add_node(
        self,
        turn_id: int,
        text: str,
        category: str,
        weight: int,
        timestamp: Optional[str] = None
    ) -> str:
        """
        Add a new memory node to the graph and connect it to relevant existing nodes.

        Args:
            turn_id: Integer identifier for the conversation turn.
            text: Packaged turn content or extracted factual text.
            category: One of 'fact', 'preference', 'relationship', 'event', 'goal'.
            weight: Importance score (1 to 5).
            timestamp: ISO 8601 timestamp string (generated automatically if None).

        Returns:
            The created node ID string (e.g., 'node_1').
        """
        node_id = f"node_{turn_id}"
        ts = timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Clean text
        clean_text = text.strip()

        # Add node with metadata attributes
        self.graph.add_node(
            node_id,
            id=node_id,
            turn_id=turn_id,
            text=clean_text,
            category=category,
            weight=int(weight),
            timestamp=ts
        )

        # 1. Connect sequential edge from the previous turn's memory node
        if self._last_node_id and self.graph.has_node(self._last_node_id):
            self.graph.add_edge(self._last_node_id, node_id, relation="FOLLOWED_BY")

        # 2. Connect semantic/category edges to previous nodes of the same category
        for other_node_id, data in self.graph.nodes(data=True):
            if other_node_id != node_id and data.get("category") == category:
                self.graph.add_edge(node_id, other_node_id, relation=f"SAME_CATEGORY_{category.upper()}")

        self._last_node_id = node_id
        self.save_to_json()
        return node_id

    def query_relevant(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant memory nodes matching keywords or substrings in query_text.

        Args:
            query_text: User message or query prompt.
            top_k: Maximum number of relevant nodes to return.

        Returns:
            List of node attribute dictionaries ordered by relevance.
        """
        if not query_text or self.graph.number_of_nodes() == 0:
            return []

        # Tokenize query into alphanumeric words (ignoring common stop words)
        stop_words = {
            "a", "an", "the", "and", "or", "in", "on", "at", "to", "for",
            "is", "are", "was", "were", "what", "where", "who", "when", "why",
            "how", "do", "does", "did", "can", "could", "would", "my", "your",
            "tell", "me", "about", "you", "i", "me", "it", "this", "that"
        }

        query_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", query_text.lower()))
        filtered_query_tokens = query_tokens - stop_words
        if not filtered_query_tokens:
            filtered_query_tokens = query_tokens

        scored_nodes = []

        for node_id, data in self.graph.nodes(data=True):
            node_text = data.get("text", "")
            node_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", node_text.lower()))

            # Keyword overlap score
            overlap = len(filtered_query_tokens & node_tokens)

            # Substring bonus: Check if full query words appear as substrings in node text
            substr_bonus = 0.0
            for token in filtered_query_tokens:
                if token in node_text.lower():
                    substr_bonus += 0.5

            # Importance weight multiplier (1 to 5)
            weight = data.get("weight", 1)
            weight_factor = 1.0 + (weight * 0.1)

            total_score = (overlap * 2.0 + substr_bonus) * weight_factor

            if total_score > 0:
                scored_nodes.append((total_score, data))

        # Sort by total score descending
        scored_nodes.sort(key=lambda x: x[0], reverse=True)

        return [item[1] for item in scored_nodes[:top_k]]

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Return a list of all nodes and their attributes."""
        return [data for _, data in self.graph.nodes(data=True)]

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the memory graph."""
        category_counts = {}
        for _, data in self.graph.nodes(data=True):
            cat = data.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "categories": category_counts
        }

    def save_to_json(self, filepath: Optional[str] = None) -> None:
        """
        Serialize the graph nodes and edges to a clean JSON file.

        Args:
            filepath: Optional override path for storage.
        """
        target_path = filepath or self.storage_path
        nodes_data = []
        for node_id, data in self.graph.nodes(data=True):
            nodes_data.append(data)

        edges_data = []
        for source, target, data in self.graph.edges(data=True):
            edges_data.append({
                "source": source,
                "target": target,
                "relation": data.get("relation", "LINK")
            })

        payload = {
            "metadata": {
                "version": "1.0",
                "total_nodes": len(nodes_data),
                "total_edges": len(edges_data)
            },
            "nodes": nodes_data,
            "edges": edges_data
        }

        try:
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[MemoryGraph] Error saving graph to {target_path}: {e}")

    def load_from_json(self, filepath: Optional[str] = None) -> bool:
        """
        Load nodes and edges from a JSON file.

        Args:
            filepath: Optional override path for storage.

        Returns:
            True if loaded successfully, False otherwise.
        """
        target_path = filepath or self.storage_path
        if not os.path.exists(target_path):
            return False

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.graph.clear()
            self._last_node_id = None

            nodes = payload.get("nodes", [])
            for node_data in nodes:
                node_id = node_data.get("id")
                if node_id:
                    self.graph.add_node(node_id, **node_data)
                    self._last_node_id = node_id

            edges = payload.get("edges", [])
            for edge_data in edges:
                src = edge_data.get("source")
                dst = edge_data.get("target")
                relation = edge_data.get("relation", "LINK")
                if src and dst and self.graph.has_node(src) and self.graph.has_node(dst):
                    self.graph.add_edge(src, dst, relation=relation)

            return True
        except Exception as e:
            print(f"[MemoryGraph] Error loading graph from {target_path}: {e}")
            return False

    def clear(self) -> None:
        """Clear all nodes and edges from memory and disk."""
        self.graph.clear()
        self._last_node_id = None
        if os.path.exists(self.storage_path):
            try:
                os.remove(self.storage_path)
            except Exception:
                pass


if __name__ == "__main__":
    # Self-test memory_graph in isolation
    print("\n--- Testing MemoryGraph in Isolation ---")
    test_graph_file = "test_graph.json"
    mem = MemoryGraph(storage_path=test_graph_file)
    mem.clear()

    # Add sample nodes
    mem.add_node(
        turn_id=1,
        text="User: I live in Seattle and work as a software engineer at CloudCorp.\nAssistant: Nice to meet you!",
        category="fact",
        weight=4
    )
    mem.add_node(
        turn_id=2,
        text="User: I strictly drink oat milk lattes and hate black coffee.\nAssistant: Oat milk lattes noted.",
        category="preference",
        weight=3
    )
    mem.add_node(
        turn_id=3,
        text="User: My brother Dave is visiting next Friday for a marathon.\nAssistant: Have fun with Dave!",
        category="relationship",
        weight=4
    )

    print(f"Graph stats after insert: {mem.get_stats()}")

    # Test retrieval
    query = "What city do I live in and what do I drink?"
    results = mem.query_relevant(query, top_k=2)
    print(f"\nQuery: '{query}'")
    print(f"Top {len(results)} matches:")
    for idx, r in enumerate(results, 1):
        print(f"  {idx}. [Category: {r['category']}, Weight: {r['weight']}] {r['text'].splitlines()[0]}")

    # Test save/reload
    mem_reloaded = MemoryGraph(storage_path=test_graph_file)
    print(f"\nReloaded graph stats: {mem_reloaded.get_stats()}")
    assert mem_reloaded.graph.number_of_nodes() == 3

    # Clean up test file
    mem.clear()
    print("\nMemoryGraph test passed successfully!")
