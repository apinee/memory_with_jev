# Memory-Graph Chatbot with TypeSafe Jev Decision Gate

A conversational AI assistant using **Google Gemini** for dialogue and **TypeSafe's Jev (System One decision model)** as the sole gatekeeper for persisting durable memories into a **NetworkX Knowledge Graph**.

---

## Architecture Overview

```
                          +-------------------------------+
                          |        User Input             |
                          +---------------+---------------+
                                          |
                                          v
                    +---------------------------------------------+
                    | 1. Memory Graph Retrieval (Keyword Overlap) |
                    |    Query existing graph.json for top-3 nodes|
                    +---------------------+-----------------------+
                                          | (Retrieved Context)
                                          v
                    +---------------------------------------------+
                    | 2. Google Gemini Conversational Agent       |
                    |    Generates contextual assistant reply     |
                    +---------------------+-----------------------+
                                          |
                                          v
                    +---------------------------------------------+
                    | 3. Turn Packaging (Plain String)            |
                    |    "User: ... \n Assistant: ..."            |
                    +---------------------+-----------------------+
                                          |
                                          v
                    +---------------------------------------------+
                    | 4. TypeSafe Jev Memory Gate (System One)    |
                    |    - should_persist (Noul: 0.0 - 1.0)       |
                    |    - category (Choice: fact|pref|goal|etc.) |
                    |    - importance (Score: 1 - 5)              |
                    +---------------------+-----------------------+
                                          |
                   +----------------------+----------------------+
                   | (prob >= 0.65 & cat != 'none')              | (prob < 0.65 or 'none')
                   v                                             v
        +-----------------------+                    +-----------------------+
        | Write Node to Graph   |                    | Discard Turn          |
        | Persist to graph.json |                    | (Ephemeral Chit-Chat) |
        +-----------------------+                    +-----------------------+
                   |
                   v
        +-----------------------+
        | visualize.py (PyVis)  |
        | Interactive HTML Map  |
        +-----------------------+
```

---

## Core Concepts Explained (Junior Engineer Friendly)

### 1. System One vs. System Two AI
* **System Two (LLMs like Gemini):** Generative models that generate text token-by-token. Powerful for conversation, but slower, probabilistic, and expensive when used just for classification.
* **System One (TypeSafe Jev):** Ultra-fast, zero-generation decision models. Instead of generating text, Jev returns typed primitives:
  * **Noul:** A binary probability (e.g., $P(\text{persist}) = 0.92$).
  * **Choice:** Categorization from a fixed list (`fact`, `preference`, `relationship`, `event`, `goal`, `none`).
  * **Score:** An ordinal rating on a 1–5 rubric for importance.

### 2. Real-World Analogy: The Executive Assistant & The Archivist
* Imagine **Gemini** is a friendly Executive Assistant having a conversation with a client in an office.
* Outside the door sits **Jev**, the Chief Archivist.
* When the client says: *"Hi, how's the weather today?"*, Jev evaluates the transcript and decides: *Ephemeral small talk $\rightarrow$ Discard.*
* When the client says: *"I only drink oat milk lattes and my brother Dave is visiting next week for the Seattle marathon"*, Jev recognizes high-value durable facts and files them into the company's permanent knowledge graph.
* On the next visit, when the client asks: *"What should we order for coffee?"*, the assistant queries the graph and immediately remembers their oat milk latte preference!

---

## Project Structure

```
├── agent.py          # Google Gemini conversational agent loop and context injection
├── jev_gate.py       # TypeSafe SDK wrapper (Noul, Choice, Score) with heuristic fallback
├── memory_graph.py   # NetworkX graph manager, JSON serializer, and keyword retrieval
├── main.py           # Interactive CLI chat loop
├── visualize.py      # Standalone PyVis HTML graph generator (memory_graph.html)
├── verify_demo.py    # Automated end-to-end multi-turn verification script
├── requirements.txt  # Python package dependencies
├── .env.example      # Environment variables template
└── graph.json        # Persisted memory graph (created automatically)
```

---

## Setup & Quickstart

### 1. Prerequisites
* Python 3.11+
* `GOOGLE_API_KEY` for Google Gemini
* `TYPESAFE_API_KEY` for TypeSafe Jev System One API

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Add your keys to `.env`:
```env
GOOGLE_API_KEY=your_gemini_key
TYPESAFE_API_KEY=your_typesafe_key
```

---

## Running the Google ADK Web Chat Interface

Launch the official Google ADK Web Chat Server:
```bash
python run_web.py
```
Or directly using ADK CLI:
```bash
python -m google.adk.cli web . --port 8000
```

1. Open your browser and navigate to: **[http://localhost:8000](http://localhost:8000)**
2. Select **`memory_app`** (`memory_graph_agent`) in the web UI.
3. Start chatting!
   * Every turn is evaluated in real time by TypeSafe's Jev decision gate (`should_persist`, `category`, `importance`).
   * Durable facts are automatically written to `graph.json`.
   * You can ask the agent to search memories or view graph stats directly in the web UI.

---

## Interactive HTML Graph Visualization

To generate and view the interactive NetworkX memory graph visualization (colored by category and sized by Jev importance):
```bash
python visualize.py --open
```
This renders `memory_graph.html` locally in your browser.

---

## Automated Verification Test

Run the full end-to-end multi-turn verification suite:
```bash
python verify_demo.py
```
