"""
TypeSafe Jev Memory Gate Module
================================
Wraps the TypeSafe Jev System One decision model to gate whether conversation
turns should be persisted into a long-term memory graph.

Key Concepts:
- System One Model: Fast, typed classification (Noul, Choice, Score) with no text generation.
- Noul: Binary probability decision (0.0 to 1.0) on long-term durability.
- Choice: Categorization into fact, preference, relationship, event, goal, or none.
- Score: Importance rating from 1 (trivial) to 5 (critical).
"""

import os
import re
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger("JevGate")

try:
    import typesafe_sdk
    from typesafe_sdk import TypeSafeClient, Noul, Choice, Score
    HAS_TYPESAFE_SDK = True
except ImportError:
    HAS_TYPESAFE_SDK = False


@dataclass
class GateDecision:
    """Represents the structured decision returned by the Jev Memory Gate."""
    should_persist: bool
    probability: float
    category: str
    importance: int
    is_mock: bool = False
    raw_data: Optional[Dict[str, Any]] = None

    def summary(self) -> str:
        status = "PERSIST" if self.should_persist else "DISCARD"
        return (
            f"[{status}] (prob={self.probability:.2f}, "
            f"category='{self.category}', importance={self.importance}/5, "
            f"mock={self.is_mock})"
        )


class JevGate:
    """
    Evaluates conversation turns using TypeSafe Jev (or a local heuristic fallback)
    to determine if the turn contains durable, high-value information for long-term memory.
    """

    CATEGORIES = ["fact", "preference", "relationship", "event", "goal", "none"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        threshold: float = 0.65,
        force_mock: bool = False
    ):
        """
        Initialize the Jev Gate.

        Args:
            api_key: TypeSafe API key. Defaults to TYPESAFE_API_KEY environment variable.
            base_url: Optional endpoint override.
            threshold: Probability cutoff for persistence (default: 0.65).
            force_mock: If True, always uses the built-in heuristic decision engine.
        """
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY", "").strip()
        custom_base = base_url or os.getenv("TYPESAFE_BASE_URL", "").strip()
        if custom_base.endswith("/v1"):
            custom_base = custom_base[:-3]
        self.base_url = custom_base or None
        self.threshold = threshold
        self.force_mock = force_mock

        # Determine if we should run in mock mode
        self.is_mock_mode = (
            self.force_mock
            or not HAS_TYPESAFE_SDK
            or not self.api_key
            or self.api_key.startswith("your_")
            or self.api_key == "placeholder"
        )

        self._client: Optional[TypeSafeClient] = None
        if not self.is_mock_mode:
            try:
                client_kwargs = {"api_key": self.api_key}
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url
                self._client = TypeSafeClient(**client_kwargs)
                logger.info("TypeSafe Jev live client initialized successfully.")
            except Exception as e:
                logger.warning(f"Failed to initialize TypeSafe client ({e}). Falling back to heuristic mock mode.")
                self.is_mock_mode = True
        else:
            logger.info("Jev Gate running in heuristic MOCK mode (no valid TYPESAFE_API_KEY detected).")

    def evaluate_turn(self, turn_text: str) -> GateDecision:
        """
        Send the turn state to Jev (or fallback) to evaluate persistence criteria.

        Args:
            turn_text: The packaged conversation turn (e.g. "User: ... \n Assistant: ...")

        Returns:
            GateDecision containing should_persist, probability, category, and importance.
        """
        if not turn_text or not turn_text.strip():
            return GateDecision(
                should_persist=False,
                probability=0.0,
                category="none",
                importance=1,
                is_mock=self.is_mock_mode
            )

        if self.is_mock_mode or not self._client:
            return self._heuristic_evaluate(turn_text)

        try:
            # Build the 3 questions for Jev System One
            questions = {
                "should_persist": Noul(
                    instructions=(
                        "Is this conversation turn durable, specific, and useful to remember in a long-term "
                        "memory graph? Answer true for user facts, personal preferences, goals, specific events, "
                        "or relationship details. Answer false for casual greetings, small talk, meta-questions, "
                        "acknowledgements, or ephemeral chit-chat."
                    )
                ),
                "category": Choice(
                    instructions="What category best describes the durable information in this turn?",
                    criteria={
                        "fact": "Specific factual data like location, workplace, job title, birthdate, or biography",
                        "preference": "Likes, dislikes, dietary habits, software tool choices, or personal tastes",
                        "relationship": "Mentions of friends, family members, pets, colleagues, or social connections",
                        "event": "Scheduled meetings, past milestones, appointments, or life events",
                        "goal": "Future plans, aspirations, targets, projects, or objectives",
                        "none": "Small talk, greetings, filler, or questions with no durable personal info"
                    }
                ),
                "importance": Score(
                    instructions="Rate the long-term importance of remembering this information on a 1 to 5 scale",
                    criteria=[
                        "Trivial or transient chit-chat with zero future value",
                        "Minor detail with low long-term relevance",
                        "Useful personal context that could assist future conversation",
                        "Important personal fact, key relationship, or explicit preference",
                        "Critical, high-value personal identity, safety, or core objective"
                    ]
                )
            }

            result = self._client.system_one(
                state=turn_text,
                questions=questions
            )

            # Parse results from typed response
            # Noul answers return probability in .noul
            prob = float(result.nouls["should_persist"].noul) if "should_persist" in result.nouls else 0.0
            
            # Choice answers return selected label in .choice
            cat = str(result.choices["category"].choice).lower() if "category" in result.choices else "none"
            if cat not in self.CATEGORIES:
                cat = "none"

            # Score answers return score level in .score
            # Score levels in rubric are 0-4 or 1-5; we normalize to 1-5 integer
            raw_score = float(result.scores["importance"].score) if "importance" in result.scores else 1.0
            importance = max(1, min(5, round(raw_score + 1 if raw_score <= 4 else raw_score)))

            should_persist = (prob >= self.threshold) and (cat != "none")

            return GateDecision(
                should_persist=should_persist,
                probability=prob,
                category=cat,
                importance=importance,
                is_mock=False,
                raw_data={
                    "should_persist_noul": prob,
                    "category_choice": cat,
                    "importance_score": importance
                }
            )

        except Exception as err:
            logger.warning(f"Jev API call encountered error: {err}. Falling back to heuristic evaluation.")
            return self._heuristic_evaluate(turn_text)

    def _heuristic_evaluate(self, turn_text: str) -> GateDecision:
        """
        High-precision heuristic simulation of Jev System One for offline/demo/fallback use.
        Extracts user intent, entity markers, and distinguishes durable statements from questions and small talk.
        """
        lines = turn_text.strip().splitlines()
        user_line = ""
        for line in lines:
            if line.lower().startswith("user:"):
                user_line = line[5:].strip()
                break
        
        target_text = user_line if user_line else turn_text
        text_lower = target_text.lower().strip()

        # 1. Questions or Meta-Queries (not new durable facts)
        question_words = [
            "what", "where", "who", "when", "why", "how", "do i", "did i", "can you",
            "could you", "tell me", "remember", "do you know", "what's my", "what is my"
        ]
        is_query = target_text.endswith("?") or any(text_lower.startswith(qw) for qw in question_words)

        # 2. Pure chit-chat / greetings / acknowledgments
        small_talk_patterns = [
            r"^(hi|hello|hey|greetings|howdy|sup)\b",
            r"\b(how are you|how's your day|how's it going|what's up|nice to meet you)\b",
            r"\b(thanks|thank you|ok|okay|cool|got it|sounds good|bye|goodbye|see ya|have a nice day)\b",
            r"\b(what can you do|who are you|help me)\b"
        ]
        is_small_talk = any(re.search(pat, text_lower) for pat in small_talk_patterns)

        if is_small_talk or is_query:
            return GateDecision(
                should_persist=False,
                probability=0.10 if is_small_talk else 0.20,
                category="none",
                importance=1,
                is_mock=True
            )

        # 3. Categorization heuristics for statements
        goal_patterns = [
            r"\b(my goal is|i plan to|i am planning to|i want to build|i aim to|i will launch|working on a project)\b",
            r"\b(target date|deadline|aspire to|hoping to achieve|dream is to)\b"
        ]
        pref_patterns = [
            r"\b(i prefer|i love|i hate|i like|i dislike|my favorite|i always use|i tend to prefer)\b",
            r"\b(i strictly|strictly drink|hate regular|favorite drink|favorite food|vegetarian|vegan|coffee|latte|tea)\b"
        ]
        rel_patterns = [
            r"\b(my wife|my husband|my son|my daughter|my brother|my sister|my father|my mother|my mom|my dad)\b",
            r"\b(my friend|my boss|my colleague|my coworker|my dog|my cat|my partner|my roommate)\b"
        ]
        event_patterns = [
            r"\b(next week|tomorrow|yesterday|on monday|on tuesday|on wednesday|on thursday|on friday|on saturday|on sunday)\b",
            r"\b(anniversary|birthday|flight|vacation|conference|wedding|graduation|appointment|meeting at|marathon)\b"
        ]
        fact_patterns = [
            r"\b(i live in|i moved to|i work at|my job is|i am a |i work as|my name is|i am from|my birthday is)\b",
            r"\b(years old|graduated from|living in|located in|citizen of|engineer at|company is)\b"
        ]

        if any(re.search(p, text_lower) for p in goal_patterns):
            cat = "goal"
            prob = 0.88
            importance = 4
        elif any(re.search(p, text_lower) for p in rel_patterns):
            cat = "relationship"
            prob = 0.85
            importance = 4
        elif any(re.search(p, text_lower) for p in pref_patterns):
            cat = "preference"
            prob = 0.82
            importance = 3
        elif any(re.search(p, text_lower) for p in event_patterns):
            cat = "event"
            prob = 0.78
            importance = 3
        elif any(re.search(p, text_lower) for p in fact_patterns):
            cat = "fact"
            prob = 0.92
            importance = 4
        elif " i " in f" {text_lower} " or text_lower.startswith("i ") or " my " in f" {text_lower} ":
            cat = "fact"
            prob = 0.70
            importance = 3
        else:
            cat = "none"
            prob = 0.15
            importance = 1

        should_persist = (prob >= self.threshold) and (cat != "none")

        return GateDecision(
            should_persist=should_persist,
            probability=prob,
            category=cat,
            importance=importance,
            is_mock=True
        )


if __name__ == "__main__":
    # Test jev_gate in isolation with sample turns
    gate = JevGate()
    print("\n--- Testing Jev Gate in Isolation ---")
    test_turns = [
        "User: Hi there! How are you today?\nAssistant: Hello! I am doing well, how can I assist you?",
        "User: I live in Seattle and work as a distributed systems engineer at CloudCorp.\nAssistant: That's great! Seattle has a thriving tech scene.",
        "User: I strictly drink oat milk lattes and hate black coffee.\nAssistant: Noted! Oat milk lattes are delicious.",
        "User: My brother Dave is visiting next Friday for a marathon.\nAssistant: Hope you and Dave have a wonderful time at the marathon!",
        "User: Thanks, that's all for now.\nAssistant: You're very welcome! Have a great day."
    ]

    for i, turn in enumerate(test_turns, 1):
        decision = gate.evaluate_turn(turn)
        print(f"\nTurn {i}:")
        print(f"Content: {turn.splitlines()[0]}")
        print(f"Decision: {decision.summary()}")
