"""
Cost Tracker Module: Real-Time Token Accounting & Unit Economics
================================================================
Calculates token usage and estimated dollar costs per request (turn)
and per full multi-turn conversation for Google Gemini and TypeSafe Jev.

Supports:
- Gemini 2.0 / 1.5 / 3.8 Flash & Pro pricing models
- Context Caching discount calculations
- TypeSafe Jev System One zero-generation decision pricing
- Token estimation fallback for offline/mock test modes
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List


# Standard Pricing per 1,000,000 (1M) tokens in USD
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gemini-3.8-flash": {
        "input_per_million": 0.10,
        "output_per_million": 0.40,
        "cached_per_million": 0.025,
    },
    "gemini-2.0-flash": {
        "input_per_million": 0.10,
        "output_per_million": 0.40,
        "cached_per_million": 0.025,
    },
    "gemini-1.5-flash": {
        "input_per_million": 0.075,
        "output_per_million": 0.30,
        "cached_per_million": 0.01875,
    },
    "gemini-1.5-pro": {
        "input_per_million": 1.25,
        "output_per_million": 5.00,
        "cached_per_million": 0.3125,
    },
    "default": {
        "input_per_million": 0.10,
        "output_per_million": 0.40,
        "cached_per_million": 0.025,
    }
}

# TypeSafe Jev System One Decision Cost per API Call
# System One models return zero text generation tokens (pure classification)
TYPESAFE_JEV_COST_PER_DECISION: float = 0.000100  # $0.0001 per call (1/100th of a cent)


@dataclass
class TokenUsage:
    """Detailed token consumption for a single turn."""
    prompt_tokens: int = 0
    candidates_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.candidates_tokens


@dataclass
class CostBreakdown:
    """Detailed financial cost for a single turn or session."""
    gemini_cost: float = 0.0
    jev_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"

    def summary(self) -> str:
        return (
            f"Total: ${self.total_cost:.6f} "
            f"(Gemini: ${self.gemini_cost:.6f}, Jev: ${self.jev_cost:.6f})"
        )


def estimate_tokens(text: str) -> int:
    """
    Fallback token estimator when usage_metadata is not returned (e.g., in mock/test mode).
    Rule of thumb: 1 token ≈ 4 characters in English.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 3.8))


def calculate_turn_cost(
    model_name: str,
    prompt_tokens: int,
    candidates_tokens: int,
    cached_tokens: int = 0,
    jev_calls: int = 1,
    is_mock: bool = False
) -> CostBreakdown:
    """
    Calculate exact cost for a single request / turn.

    Args:
        model_name: The Gemini model identifier.
        prompt_tokens: Number of prompt/input tokens.
        candidates_tokens: Number of completion/output tokens.
        cached_tokens: Tokens read from Gemini context cache (discounted).
        jev_calls: Number of Jev System One decisions made in this turn.
        is_mock: If True, calculates theoretical cost as if live API was called.
    """
    pricing = MODEL_PRICING.get(model_name, MODEL_PRICING["default"])

    # Uncached prompt tokens
    billable_prompt_tokens = max(0, prompt_tokens - cached_tokens)

    # Gemini cost computation
    gemini_input_cost = (billable_prompt_tokens / 1_000_000.0) * pricing["input_per_million"]
    gemini_cached_cost = (cached_tokens / 1_000_000.0) * pricing["cached_per_million"]
    gemini_output_cost = (candidates_tokens / 1_000_000.0) * pricing["output_per_million"]
    gemini_total = gemini_input_cost + gemini_cached_cost + gemini_output_cost

    # Jev System One decision cost
    jev_total = jev_calls * TYPESAFE_JEV_COST_PER_DECISION

    return CostBreakdown(
        gemini_cost=round(gemini_total, 8),
        jev_cost=round(jev_total, 8),
        total_cost=round(gemini_total + jev_total, 8)
    )


@dataclass
class ConversationCostSummary:
    """Cumulative metrics and cost breakdown for an entire conversation."""
    turn_count: int = 0
    total_prompt_tokens: int = 0
    total_candidates_tokens: int = 0
    total_cached_tokens: int = 0
    total_tokens: int = 0
    total_gemini_cost: float = 0.0
    total_jev_cost: float = 0.0
    total_cost: float = 0.0
    naive_unpruned_tokens: int = 0
    tokens_saved: int = 0
    cost_saved: float = 0.0

    def generate_report(self) -> str:
        avg_cost_per_turn = (self.total_cost / self.turn_count) if self.turn_count > 0 else 0.0
        avg_tokens_per_turn = (self.total_tokens / self.turn_count) if self.turn_count > 0 else 0.0
        savings_pct = (
            (self.tokens_saved / self.naive_unpruned_tokens * 100.0)
            if self.naive_unpruned_tokens > 0 else 0.0
        )

        return (
            "=================================================================\n"
            "                 CONVERSATION COST & TOKEN REPORT                \n"
            "=================================================================\n"
            f"Turns Processed       : {self.turn_count}\n"
            f"Total Tokens Used     : {self.total_tokens:,} "
            f"(Prompt: {self.total_prompt_tokens:,}, Output: {self.total_candidates_tokens:,})\n"
            f"Avg Tokens / Request  : {avg_tokens_per_turn:.1f} tokens\n"
            "-----------------------------------------------------------------\n"
            f"Gemini LLM Cost       : ${self.total_gemini_cost:.6f} USD\n"
            f"TypeSafe Jev Cost     : ${self.total_jev_cost:.6f} USD\n"
            f"Total Conversation    : ${self.total_cost:.6f} USD\n"
            f"Avg Cost per Request  : ${avg_cost_per_turn:.6f} USD (~{avg_cost_per_turn*100:.4f} cents)\n"
            "-----------------------------------------------------------------\n"
            "EFFICIENCY SAVINGS (Memory Graph vs Naive Full-History Injection):\n"
            f"Naive Tokens Needed   : {self.naive_unpruned_tokens:,} tokens\n"
            f"Tokens Saved by Graph : {self.tokens_saved:,} tokens ({savings_pct:.1f}% reduction)\n"
            f"Cost Saved by Graph   : ${self.cost_saved:.6f} USD\n"
            "================================================================="
        )


class SessionCostTracker:
    """Tracks token consumption and running financial cost across a multi-turn chat."""

    def __init__(self, model_name: str = "gemini-3.8-flash"):
        self.model_name = model_name
        self.turn_records: List[Dict[str, Any]] = []
        self._cumulative_raw_history_tokens: int = 0

    def record_turn(
        self,
        user_message: str,
        assistant_reply: str,
        token_usage: TokenUsage,
        jev_decision_made: bool = True
    ) -> CostBreakdown:
        """Record turn metrics and return financial cost breakdown."""
        cost = calculate_turn_cost(
            model_name=self.model_name,
            prompt_tokens=token_usage.prompt_tokens,
            candidates_tokens=token_usage.candidates_tokens,
            cached_tokens=token_usage.cached_tokens,
            jev_calls=1 if jev_decision_made else 0
        )

        # Track what a naive conversation without graph would have cost
        # (Where every turn appends to history and sends entire conversation history)
        turn_raw_tokens = estimate_tokens(user_message) + estimate_tokens(assistant_reply)
        self._cumulative_raw_history_tokens += turn_raw_tokens
        naive_prompt_tokens = self._cumulative_raw_history_tokens + 150  # base system instruction

        self.turn_records.append({
            "turn_index": len(self.turn_records) + 1,
            "token_usage": token_usage,
            "cost": cost,
            "naive_prompt_tokens": naive_prompt_tokens
        })
        return cost

    def get_summary(self) -> ConversationCostSummary:
        """Compute aggregate summary and savings metrics."""
        summary = ConversationCostSummary(turn_count=len(self.turn_records))

        for rec in self.turn_records:
            u: TokenUsage = rec["token_usage"]
            c: CostBreakdown = rec["cost"]
            naive_prompt = rec["naive_prompt_tokens"]

            summary.total_prompt_tokens += u.prompt_tokens
            summary.total_candidates_tokens += u.candidates_tokens
            summary.total_cached_tokens += u.cached_tokens
            summary.total_tokens += u.total_tokens

            summary.total_gemini_cost += c.gemini_cost
            summary.total_jev_cost += c.jev_cost
            summary.total_cost += c.total_cost

            summary.naive_unpruned_tokens += (naive_prompt + u.candidates_tokens)

        summary.tokens_saved = max(0, summary.naive_unpruned_tokens - summary.total_tokens)
        pricing = MODEL_PRICING.get(self.model_name, MODEL_PRICING["default"])
        summary.cost_saved = (summary.tokens_saved / 1_000_000.0) * pricing["input_per_million"]

        return summary
