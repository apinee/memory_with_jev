"""
Memory Graph Visualizer
=======================
Renders the memory graph from `graph.json` into an interactive HTML visualization
using PyVis. Nodes are color-coded by category and scaled by importance weight.

Color Scheme:
- Fact:         Blue (#3B82F6)
- Preference:   Emerald Green (#10B981)
- Relationship: Purple (#8B5CF6)
- Event:        Amber (#F59E0B)
- Goal:         Rose Red (#EF4444)
- Default:      Slate Gray (#64748B)
"""

import os
import json
import argparse
from typing import Optional

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False


CATEGORY_COLORS = {
    "fact": {
        "background": "#3B82F6",
        "border": "#1D4ED8",
        "highlight": {"background": "#60A5FA", "border": "#1E40AF"}
    },
    "preference": {
        "background": "#10B981",
        "border": "#047857",
        "highlight": {"background": "#34D399", "border": "#065F46"}
    },
    "relationship": {
        "background": "#8B5CF6",
        "border": "#6D28D9",
        "highlight": {"background": "#A78BFA", "border": "#5B21B6"}
    },
    "event": {
        "background": "#F59E0B",
        "border": "#B45309",
        "highlight": {"background": "#FBBF24", "border": "#92400E"}
    },
    "goal": {
        "background": "#EF4444",
        "border": "#B91C1C",
        "highlight": {"background": "#F87171", "border": "#991B1B"}
    },
    "default": {
        "background": "#64748B",
        "border": "#334155",
        "highlight": {"background": "#94A3B8", "border": "#1E293B"}
    }
}


def render_graph_html(
    json_path: str = "graph.json",
    output_html: str = "memory_graph.html",
    height: str = "750px",
    width: str = "100%",
    open_browser: bool = False
) -> Optional[str]:
    """
    Generate an interactive HTML visualization of the memory graph.

    Args:
        json_path: Path to the persisted graph JSON file.
        output_html: Output HTML filename.
        height: Canvas height.
        width: Canvas width.
        open_browser: Whether to launch the browser automatically.

    Returns:
        Absolute path to the generated HTML file if successful, or None.
    """
    if not os.path.exists(json_path):
        print(f"[-] No graph file found at '{json_path}'. Run chat sessions first to populate memories.")
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[-] Failed to read {json_path}: {e}")
        return None

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    if not nodes:
        print("[!] Graph is currently empty (0 nodes). Chat with the bot to build memories first.")
        return None

    if not HAS_PYVIS:
        print("[-] 'pyvis' is not installed. Please run 'pip install pyvis' to generate HTML visualizations.")
        return None

    # Initialize PyVis network with dark mode styling
    net = Network(
        height=height,
        width=width,
        bgcolor="#0F172A",
        font_color="#F8FAFC",
        directed=True
    )

    # Physics configuration for smooth spring layout
    net.barnes_hut(
        gravity=-3500,
        central_gravity=0.3,
        spring_length=160,
        spring_strength=0.05,
        damping=0.09
    )

    # Add Nodes
    for node in nodes:
        node_id = node.get("id")
        turn_id = node.get("turn_id", "?")
        category = node.get("category", "default").lower()
        weight = node.get("weight", 1)
        text = node.get("text", "")
        ts = node.get("timestamp", "N/A")

        # Color mapping
        color_scheme = CATEGORY_COLORS.get(category, CATEGORY_COLORS["default"])

        # Node size scaled by importance weight (1 to 5)
        node_size = 18 + (weight * 7)

        # Truncated label for clean canvas
        first_line = text.splitlines()[0] if text else "Memory"
        if len(first_line) > 36:
            label = f"{first_line[:33]}..."
        else:
            label = first_line

        # Rich interactive HTML tooltip
        formatted_text = text.replace("\n", "<br>")
        title_tooltip = (
            f"<div style='font-family: sans-serif; padding: 6px; font-size: 13px; max-width: 320px;'>"
            f"<b>Node:</b> {node_id} (Turn #{turn_id})<br>"
            f"<b>Category:</b> <span style='color:{color_scheme['background']}; font-weight:bold;'>{category.upper()}</span><br>"
            f"<b>Importance:</b> {'★' * weight}{'☆' * (5 - weight)} ({weight}/5)<br>"
            f"<b>Timestamp:</b> {ts}<br><hr style='border: 0.5px solid #475569; margin: 6px 0;'>"
            f"<b>Content:</b><br>{formatted_text}"
            f"</div>"
        )

        net.add_node(
            n_id=node_id,
            label=f"[{category.upper()}] {label}",
            title=title_tooltip,
            color=color_scheme,
            size=node_size,
            shape="dot",
            font={"size": 13, "color": "#F8FAFC", "face": "Inter, sans-serif"}
        )

    # Add Edges
    for edge in edges:
        src = edge.get("source")
        dst = edge.get("target")
        relation = edge.get("relation", "LINK")

        if src and dst:
            edge_color = "#38BDF8" if "FOLLOWED_BY" in relation else "#818CF8"
            edge_dashes = False if "FOLLOWED_BY" in relation else True
            net.add_edge(
                source=src,
                to=dst,
                title=f"Relation: {relation}",
                label=relation.lower().replace("_", " "),
                color={"color": edge_color, "highlight": "#F43F5E"},
                dashes=edge_dashes,
                arrows="to",
                width=1.5
            )

    # Custom HTML header injection for aesthetic legend
    legend_html = """
    <div style="position: absolute; top: 15px; left: 15px; background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(8px); padding: 14px 18px; border-radius: 10px;
                border: 1px solid #334155; font-family: Inter, system-ui, sans-serif;
                color: #F8FAFC; z-index: 1000; box-shadow: 0 10px 25px rgba(0,0,0,0.5);">
        <h3 style="margin: 0 0 10px 0; font-size: 15px; font-weight: 700; letter-spacing: 0.5px;">Memory Graph Categories</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px;">
            <div><span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:#3B82F6; margin-right:6px;"></span><b>Fact</b></div>
            <div><span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:#10B981; margin-right:6px;"></span><b>Preference</b></div>
            <div><span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:#8B5CF6; margin-right:6px;"></span><b>Relationship</b></div>
            <div><span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:#F59E0B; margin-right:6px;"></span><b>Event</b></div>
            <div><span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:#EF4444; margin-right:6px;"></span><b>Goal</b></div>
        </div>
        <div style="margin-top: 10px; font-size: 11px; color: #94A3B8; border-top: 1px solid #334155; padding-top: 6px;">
            Node size represents Jev Importance Score (1-5)
        </div>
    </div>
    """

    abs_output_path = os.path.abspath(output_html)
    net.write_html(abs_output_path)

    # Post-process generated HTML (add lang attribute for a11y & inject legend)
    try:
        with open(abs_output_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Fix missing lang attribute on <html>
        if "<html lang=" not in html_content:
            html_content = html_content.replace("<html>", '<html lang="en">', 1)

        # Insert legend right after <body>
        if "<body>" in html_content:
            html_content = html_content.replace("<body>", f"<body>\n{legend_html}", 1)

        with open(abs_output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as e:
        print(f"[!] Warning: Could not post-process generated HTML: {e}")

    print(f"[+] Memory graph visualization successfully generated at: {abs_output_path}")

    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{abs_output_path}")

    return abs_output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render the memory graph into an interactive HTML visualization.")
    parser.add_argument("--json", default="graph.json", help="Path to graph JSON file (default: graph.json)")
    parser.add_argument("--output", default="memory_graph.html", help="Path to output HTML file (default: memory_graph.html)")
    parser.add_argument("--open", action="store_true", help="Open generated HTML in default browser")

    args = parser.parse_args()
    render_graph_html(
        json_path=args.json,
        output_html=args.output,
        open_browser=args.open
    )
