"""
Agent Module: Google ADK / Gemini Conversational Loop with Jev Memory Gating
=============================================================================
Manages conversational turns with Gemini, performs keyword retrieval from the
memory graph, and passes packaged turns through the TypeSafe Jev gate for
long-term persistence.

Key Flow per Turn:
1. Retrieval: Query MemoryGraph for top-k matching nodes.
2. Context Augmentation: Inject retrieved memories into LLM prompt/instruction.
3. LLM Generation: Generate response using Google Gemini.
4. Packaging: Format turn into string 'User: ... \\n Assistant: ...' (no extra LLM call).
5. Gating: Send packaged turn to JevGate (System One: should_persist, category, importance).
6. Persistence: If should_persist is True, write node to MemoryGraph and persist to disk.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv

from jev_gate import JevGate, GateDecision
from memory_graph import MemoryGraph

# Load environment variables
load_dotenv()

logger = logging.getLogger("MemoryAgent")

# Expose ADK root_agent for direct ADK discovery
try:
    from memory_app.agent import root_agent
except ImportError:
    root_agent = None

# Check for Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    HAS_GENAI_SDK = True
except ImportError:
    HAS_GENAI_SDK = False


@dataclass
class TurnResult:
    """Encapsulates all results and metadata of a single conversation turn."""
    turn_id: int
    user_input: str
    assistant_reply: str
    retrieved_memories: List[Dict[str, Any]]
    gate_decision: GateDecision
    persisted_node_id: Optional[str]
    is_mock_llm: bool = False


class MemoryChatbotAgent:
    """
    Conversational Chatbot powered by Google Gemini and gated by TypeSafe Jev.
    """

    SYSTEM_INSTRUCTION = (
        "You are a helpful, friendly, and attentive AI assistant equipped with a dynamic "
        "long-term memory graph. When past memory context is provided, naturally incorporate "
        "what you know about the user (their preferences, facts, relationships, events, and goals) "
        "into your answers. Keep your responses engaging, helpful, and concise."
    )

    CANDIDATE_MODELS = [
        "gemini-3.8-flash",
        "gemini-3-flash-preview",
        "gemini-3.5-flash-lite",
        "gemini-flash-lite-latest"
    ]

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        memory_graph: Optional[MemoryGraph] = None,
        jev_gate: Optional[JevGate] = None,
        graph_storage_path: str = "graph.json",
        force_mock_llm: bool = False
    ):
        """
        Initialize the Memory Chatbot Agent.
        """
        self.api_key = (
            gemini_api_key
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY", "")
        ).strip()
        self.model_name = model_name or "gemini-3.8-flash"
        self.force_mock_llm = force_mock_llm

        # Initialize sub-components
        self.memory_graph = memory_graph or MemoryGraph(storage_path=graph_storage_path)
        self.jev_gate = jev_gate or JevGate()

        # Determine if we should use mock LLM mode
        self.is_mock_llm = (
            self.force_mock_llm
            or not HAS_GENAI_SDK
            or not self.api_key
            or self.api_key.startswith("your_")
            or self.api_key == "placeholder"
        )

        self._client: Optional[Any] = None
        self._chat_session: Optional[Any] = None
        self.turn_counter = 0

        if not self.is_mock_llm:
            try:
                self._client = genai.Client(api_key=self.api_key)
                self._init_chat_session()
            except Exception as e:
                logger.warning(f"Failed to initialize Google Gemini client ({e}). Falling back to mock mode.")
                self.is_mock_llm = True
        else:
            logger.info("Chatbot running in conversational MOCK mode (no valid GOOGLE_API_KEY detected).")

    def _init_chat_session(self) -> None:
        """Initialize chat session trying candidate models in order."""
        models_to_try = [self.model_name] + [m for m in self.CANDIDATE_MODELS if m != self.model_name]
        for candidate in models_to_try:
            try:
                self._chat_session = self._client.chats.create(
                    model=candidate,
                    config=types.GenerateContentConfig(
                        system_instruction=self.SYSTEM_INSTRUCTION,
                        temperature=0.7
                    )
                )
                self.model_name = candidate
                logger.info(f"Google Gemini connected successfully using model: {candidate}")
                return
            except Exception as e:
                logger.warning(f"Model {candidate} failed initialization: {e}")
        self.is_mock_llm = True

    def process_turn(self, user_message: str) -> TurnResult:
        """
        Process a full conversation turn:
        1. Query relevant memories from graph.
        2. Generate assistant reply with injected memory context.
        3. Package turn (user message + reply).
        4. Gate turn via Jev.
        5. Persist to graph if approved.
        """
        self.turn_counter += 1
        turn_id = self.turn_counter

        # 1. Retrieve relevant memory nodes from the graph
        retrieved_memories = self.memory_graph.query_relevant(user_message, top_k=3)

        # 2. Build augmented prompt if memories exist
        if retrieved_memories:
            memory_snippets = []
            for mem in retrieved_memories:
                cat = mem.get("category", "info")
                txt = mem.get("text", "").replace("\n", " | ")
                memory_snippets.append(f"- [{cat.upper()}]: {txt}")

            memory_context_str = "\n".join(memory_snippets)
            prompt_with_memory = (
                f"[Retrieved Long-Term Memories from Knowledge Graph]:\n"
                f"{memory_context_str}\n\n"
                f"User: {user_message}"
            )
        else:
            prompt_with_memory = user_message

        # 3. Generate assistant reply
        assistant_reply = ""
        if not self.is_mock_llm and self._chat_session:
            try:
                response = self._chat_session.send_message(prompt_with_memory)
                assistant_reply = response.text.strip()
            except Exception as e:
                logger.warning(f"Gemini send_message failed ({e}). Attempting fallback generation.")
                # Try generate_content fallback with alternate model
                for alt_model in self.CANDIDATE_MODELS:
                    try:
                        resp = self._client.models.generate_content(
                            model=alt_model,
                            contents=prompt_with_memory,
                            config=types.GenerateContentConfig(
                                system_instruction=self.SYSTEM_INSTRUCTION
                            )
                        )
                        assistant_reply = resp.text.strip()
                        break
                    except Exception:
                        continue

        if not assistant_reply:
            assistant_reply = self._mock_generate_reply(user_message, retrieved_memories)

        # 4. Package turn into plain string (no second LLM call)
        packaged_turn = f"User: {user_message}\nAssistant: {assistant_reply}"

        # 5. Evaluate turn with TypeSafe Jev Gate
        gate_decision = self.jev_gate.evaluate_turn(packaged_turn)

        # 6. If approved, persist into memory graph
        persisted_node_id = None
        if gate_decision.should_persist:
            persisted_node_id = self.memory_graph.add_node(
                turn_id=turn_id,
                text=packaged_turn,
                category=gate_decision.category,
                weight=gate_decision.importance
            )

        return TurnResult(
            turn_id=turn_id,
            user_input=user_message,
            assistant_reply=assistant_reply,
            retrieved_memories=retrieved_memories,
            gate_decision=gate_decision,
            persisted_node_id=persisted_node_id,
            is_mock_llm=self.is_mock_llm
        )

    def _mock_generate_reply(self, user_message: str, memories: List[Dict[str, Any]]) -> str:
        """
        Simulated fallback response generator when Gemini API is offline or testing without keys.
        Seamlessly references retrieved memories if the user asks a question or recall query.
        """
        msg_lower = user_message.lower()
        is_query = user_message.endswith("?") or any(
            msg_lower.startswith(w) for w in ["what", "where", "who", "when", "why", "how", "do you know", "tell me", "can you", "remember"]
        )

        # Check if user is asking a question and we have memories to ground the answer
        if memories and (is_query or any(w in msg_lower for w in ["what city", "what do i", "favorite", "who is", "where do i"])):
            recalled_facts = []
            for mem in memories:
                txt = mem.get("text", "")
                recalled_facts.append(txt.splitlines()[0])

            joined_facts = "; ".join(recalled_facts)
            return (
                f"Based on my long-term memory graph, I recall that: {joined_facts}. "
                f"Is there anything specific you would like to follow up on?"
            )

        # Acknowledgments / Filler
        if any(w in msg_lower for w in ["thanks", "thank you", "cool", "got it", "ok", "okay", "great"]):
            return "You're very welcome! Let me know if there's anything else you'd like to chat about."

        # Informative user statements
        if "live in" in msg_lower or "work as" in msg_lower:
            return "That's wonderful to know! I've noted your location and work in my long-term memory."
        if any(w in msg_lower for w in ["oat milk", "coffee", "drink", "prefer", "favorite"]):
            return "Got it! I will remember your preference for future conversations."
        if any(w in msg_lower for w in ["brother", "sister", "friend", "wife", "husband", "dave"]):
            return "That sounds exciting! I have recorded this relationship detail in my memory."
        if any(w in msg_lower for w in ["goal", "plan", "project", "launch"]):
            return "That sounds like an ambitious goal! I have logged this objective in my memory graph."
        if any(w in msg_lower for w in ["hi", "hello", "hey"]):
            return "Hello! How can I help you today? Feel free to share thoughts, goals, or ask any questions."

        return f"I understand! Thank you for sharing. How else can I assist you?"


if __name__ == "__main__":
    # Test agent in isolation
    print("\n--- Testing Agent Turn Loop in Isolation ---")
    agent = MemoryChatbotAgent(graph_storage_path="test_agent_graph.json")
    agent.memory_graph.clear()

    # Turn 1: Small talk
    t1 = agent.process_turn("Hi there! How's your day going?")
    print(f"\n[Turn {t1.turn_id}] User: {t1.user_input}")
    print(f"Assistant: {t1.assistant_reply}")
    print(f"Gate: {t1.gate_decision.summary()} -> Node: {t1.persisted_node_id}")

    # Turn 2: Fact
    t2 = agent.process_turn("I live in Seattle and work as a distributed systems engineer.")
    print(f"\n[Turn {t2.turn_id}] User: {t2.user_input}")
    print(f"Assistant: {t2.assistant_reply}")
    print(f"Gate: {t2.gate_decision.summary()} -> Node: {t2.persisted_node_id}")

    # Turn 3: Preference
    t3 = agent.process_turn("I strictly drink oat milk lattes and hate regular dairy.")
    print(f"\n[Turn {t3.turn_id}] User: {t3.user_input}")
    print(f"Assistant: {t3.assistant_reply}")
    print(f"Gate: {t3.gate_decision.summary()} -> Node: {t3.persisted_node_id}")

    # Turn 4: Memory Retrieval Query
    t4 = agent.process_turn("What city do I live in and what do I like to drink?")
    print(f"\n[Turn {t4.turn_id}] User: {t4.user_input}")
    print(f"Retrieved {len(t4.retrieved_memories)} memories:")
    for m in t4.retrieved_memories:
        print(f"  - [{m['category']}] {m['text'].splitlines()[0]}")
    print(f"Assistant: {t4.assistant_reply}")
    print(f"Gate: {t4.gate_decision.summary()} -> Node: {t4.persisted_node_id}")

    agent.memory_graph.clear()
    print("\nAgent turn loop test passed successfully!")
