"""
Comprehensive Evaluation & Cost Benchmark Suite
=================================================
Runs an end-to-end evaluation of the Memory-Graph Chatbot with TypeSafe Jev Gate.

Evaluates 3 Core Dimensions:
1. System One Memory Gate:
   - Accuracy, Precision, Recall, F1-Score (Persist vs. Discard)
   - Category Classification Accuracy (fact, preference, goal, relationship, etc.)
2. Memory Graph Retrieval:
   - Hit Rate / Recall@k for subsequent context questions
3. Token Consumption & Financial Cost:
   - Cost per request (turn) & total cost per conversation
   - Measured comparison against naive unpruned chat history
"""

import sys
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from agent import MemoryChatbotAgent, TurnResult
from memory_graph import MemoryGraph
from jev_gate import JevGate
from cost_tracker import ConversationCostSummary


@dataclass
class EvalTestCase:
    """Represents a single evaluated turn in the test suite."""
    turn_id: int
    label: str
    user_input: str
    expected_persist: bool
    expected_category: str
    is_recall_query: bool = False
    expected_recalled_keywords: Optional[List[str]] = None


# Curated Gold-Standard Evaluation Dataset
BENCHMARK_DATASET: List[EvalTestCase] = [
    EvalTestCase(
        turn_id=1,
        label="Greeting / Small Talk",
        user_input="Hello! Good morning, how are you today?",
        expected_persist=False,
        expected_category="none"
    ),
    EvalTestCase(
        turn_id=2,
        label="Personal Fact (Location & Career)",
        user_input="I live in Seattle and work as a distributed systems engineer at CloudCorp.",
        expected_persist=True,
        expected_category="fact"
    ),
    EvalTestCase(
        turn_id=3,
        label="Personal Dietary Preference",
        user_input="I strictly drink oat milk lattes and dislike whole milk.",
        expected_persist=True,
        expected_category="preference"
    ),
    EvalTestCase(
        turn_id=4,
        label="Family Relationship & Upcoming Event",
        user_input="My brother Dave is visiting next week for the Seattle marathon.",
        expected_persist=True,
        expected_category="relationship"
    ),
    EvalTestCase(
        turn_id=5,
        label="Professional Goal",
        user_input="My goal is to launch an open source vector database project by December.",
        expected_persist=True,
        expected_category="goal"
    ),
    EvalTestCase(
        turn_id=6,
        label="Ephemeral Acknowledgment",
        user_input="Awesome, that sounds great. Thank you so much!",
        expected_persist=False,
        expected_category="none"
    ),
    EvalTestCase(
        turn_id=7,
        label="Filler Chit-Chat",
        user_input="Is it raining outside today?",
        expected_persist=False,
        expected_category="none"
    ),
    EvalTestCase(
        turn_id=8,
        label="Multi-Fact Retrieval Query",
        user_input="What city do I live in, what is my favorite drink, and who is visiting me?",
        expected_persist=False,
        expected_category="none",
        is_recall_query=True,
        expected_recalled_keywords=["seattle", "oat milk", "dave"]
    )
]


class EvaluationRunner:
    """Orchestrates evaluation runs and calculates metrics."""

    def __init__(self, storage_path: str = "eval_graph.json"):
        self.storage_path = storage_path
        self.memory_graph = MemoryGraph(storage_path=self.storage_path)
        self.jev_gate = JevGate()
        self.agent = MemoryChatbotAgent(
            memory_graph=self.memory_graph,
            jev_gate=self.jev_gate,
            graph_storage_path=self.storage_path
        )

    def run_eval(self) -> Dict[str, Any]:
        """Execute the benchmark suite and compute metrics."""
        self.memory_graph.clear()

        print("=================================================================")
        print("    RUNNING EVALUATION & COST BENCHMARK (MEMORY CHATBOT AGENT)   ")
        print("=================================================================")
        print(f"Model Under Test   : {self.agent.model_name}")
        print(f"Gatekeeper Model   : TypeSafe Jev (System One)")
        print(f"Total Test Turns   : {len(BENCHMARK_DATASET)}")
        print("-----------------------------------------------------------------\n")

        tp = 0  # True Positive (Expected persist, persisted)
        fp = 0  # False Positive (Expected discard, persisted)
        tn = 0  # True Negative (Expected discard, discarded)
        fn = 0  # False Negative (Expected persist, discarded)

        correct_categories = 0
        persisted_turns_count = 0

        retrieval_queries_count = 0
        retrieval_hits = 0

        results: List[Dict[str, Any]] = []

        for tc in BENCHMARK_DATASET:
            turn_res: TurnResult = self.agent.process_turn(tc.user_input)
            actual_persist = turn_res.gate_decision.should_persist
            actual_cat = turn_res.gate_decision.category.lower()

            # 1. Gate Confusion Matrix
            if tc.expected_persist and actual_persist:
                tp += 1
            elif not tc.expected_persist and not actual_persist:
                tn += 1
            elif not tc.expected_persist and actual_persist:
                fp += 1
            elif tc.expected_persist and not actual_persist:
                fn += 1

            # 2. Category Accuracy
            if actual_persist:
                persisted_turns_count += 1
                if actual_cat == tc.expected_category.lower():
                    correct_categories += 1

            # 3. Retrieval Recall check
            hit_success = False
            if tc.is_recall_query:
                retrieval_queries_count += 1
                retrieved_texts = " ".join([
                    m.get("text", "").lower() for m in turn_res.retrieved_memories
                ])
                if tc.expected_recalled_keywords:
                    hits = sum(
                        1 for kw in tc.expected_recalled_keywords
                        if kw.lower() in retrieved_texts
                    )
                    hit_rate = hits / len(tc.expected_recalled_keywords)
                    if hit_rate >= 0.65:
                        retrieval_hits += 1
                        hit_success = True

            token_u = turn_res.token_usage
            cost_u = turn_res.cost

            results.append({
                "turn_id": tc.turn_id,
                "label": tc.label,
                "user_input": tc.user_input,
                "actual_persist": actual_persist,
                "expected_persist": tc.expected_persist,
                "category": actual_cat,
                "expected_category": tc.expected_category,
                "tokens": token_u.total_tokens if token_u else 0,
                "turn_cost": cost_u.total_cost if cost_u else 0.0,
                "retrieval_hit": hit_success if tc.is_recall_query else None
            })

            # Real-time console line
            status_symbol = "PERSIST" if actual_persist else "DISCARD"
            match_symbol = "PASS" if (actual_persist == tc.expected_persist) else "FAIL"
            cost_str = f"${cost_u.total_cost:.6f}" if cost_u else "$0.00"
            print(
                f"Turn {tc.turn_id:02d} [{match_symbol}] | Gate: {status_symbol:<7} "
                f"({turn_res.gate_decision.probability:.2f}) | "
                f"Cat: {actual_cat:<12} | Cost: {cost_str} | "
                f"Msg: '{tc.user_input[:40]}...'"
            )

        # Compute Gate Metrics
        total_tests = len(BENCHMARK_DATASET)
        accuracy = (tp + tn) / total_tests if total_tests > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        cat_accuracy = (
            correct_categories / persisted_turns_count if persisted_turns_count > 0 else 0.0
        )
        retrieval_hit_rate = (
            retrieval_hits / retrieval_queries_count if retrieval_queries_count > 0 else 1.0
        )

        cost_summary: ConversationCostSummary = self.agent.cost_tracker.get_summary()

        print("\n" + "=" * 65)
        print("                      EVALUATION SCORECARD                       ")
        print("=" * 65)
        print(f"1. MEMORY GATE PERFORMANCE (System One):")
        print(f"   - Total Turns Tested      : {total_tests}")
        print(f"   - Accuracy                : {accuracy * 100:.1f}%")
        print(f"   - Precision               : {precision * 100:.1f}% (TP={tp}, FP={fp})")
        print(f"   - Recall                  : {recall * 100:.1f}% (FN={fn})")
        print(f"   - F1 Score                : {f1_score:.3f}")
        print(f"   - Category Match Rate     : {cat_accuracy * 100:.1f}%")
        print(f"   - Ephemeral Small Talk TN : {tn} turns correctly discarded")
        print("-----------------------------------------------------------------")
        print(f"2. MEMORY GRAPH RETRIEVAL ACCURACY:")
        print(f"   - Recall Queries Tested   : {retrieval_queries_count}")
        print(f"   - Retrieval Hit Rate @ 3  : {retrieval_hit_rate * 100:.1f}%")
        print("-----------------------------------------------------------------")
        print(cost_summary.generate_report())

        # Cleanup test graph file
        if os.path.exists(self.storage_path):
            try:
                os.remove(self.storage_path)
            except Exception:
                pass

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "category_accuracy": cat_accuracy,
            "retrieval_hit_rate": retrieval_hit_rate,
            "cost_summary": cost_summary
        }


def main():
    runner = EvaluationRunner()
    metrics = runner.run_eval()
    if metrics["accuracy"] >= 0.85 and metrics["retrieval_hit_rate"] >= 0.80:
        print("\n[+] EVALUATION RESULT: SUITE PASSED EXCELLENTLY!\n")
        sys.exit(0)
    else:
        print("\n[-] EVALUATION RESULT: BELOW THRESHOLD\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
