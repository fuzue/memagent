"""Export the topic graph as a self-contained HTML file with D3 force layout.

Colors are derived deterministically from the type string via hashing — any new
node or edge type the extractor invents gets a stable, unique color with no
code changes required.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .db import get_conn, init_db

# Reserved colors only for the structural types (episode is not user-extracted)
RESERVED_NODE_COLORS = {
    "episode": "#475569",
}


def _color_for(label: str, *, lightness: int = 55, saturation: int = 65) -> str:
    """Hash a label to a deterministic HSL color."""
    if label in RESERVED_NODE_COLORS:
        return RESERVED_NODE_COLORS[label]
    h = int(hashlib.md5(label.lower().encode()).hexdigest()[:6], 16) % 360
    return f"hsl({h}, {saturation}%, {lightness}%)"


def export_html(output_path: Path) -> None:
    init_db()

    import time as _time
    with get_conn() as conn:
        node_rows = conn.execute(
            "SELECT id, csum, craw, entity_type FROM topic_nodes"
        ).fetchall()
        edge_rows = conn.execute(
            "SELECT source_id, target_id, edge_type, weight FROM topic_edges"
        ).fetchall()
        episode_rows = conn.execute(
            "SELECT id, summary, started_at FROM episodes ORDER BY started_at DESC LIMIT 50"
        ).fetchall()
        link_rows = conn.execute(
            "SELECT entity_node_id, episode_id FROM entity_episode_links"
        ).fetchall()

    nodes = [
        {
            "id": r["id"],
            "label": r["csum"].split(":")[0].strip(),
            "summary": r["csum"],
            "detail": r["craw"],
            "type": r["entity_type"],
            "shape": "circle",
            "color": _color_for(r["entity_type"] or "fact"),
        }
        for r in node_rows
    ]

    # Episodes as a different node shape (smaller, square)
    for r in episode_rows:
        date_str = _time.strftime("%m-%d %H:%M", _time.localtime(r["started_at"]))
        nodes.append({
            "id": r["id"],
            "label": date_str,
            "summary": f"Episode {date_str}",
            "detail": r["summary"] or "(no summary)",
            "type": "episode",
            "shape": "square",
            "color": _color_for("episode"),
        })

    node_ids = {n["id"] for n in nodes}
    edges = [
        {
            "source": r["source_id"],
            "target": r["target_id"],
            "type": r["edge_type"],
            "color": _color_for(r["edge_type"], lightness=50),
            "dashed": False,
        }
        for r in edge_rows
        if r["source_id"] in node_ids and r["target_id"] in node_ids
    ]

    # Cross-links: entity ↔ episode (rendered dashed)
    for r in link_rows:
        if r["entity_node_id"] in node_ids and r["episode_id"] in node_ids:
            edges.append({
                "source": r["entity_node_id"],
                "target": r["episode_id"],
                "type": "MENTIONED_IN",
                "color": _color_for("MENTIONED_IN", lightness=50),
                "dashed": True,
            })

    graph_data = json.dumps({"nodes": nodes, "edges": edges})

    d3_cache = Path.home() / ".memagent" / "d3.v7.min.js"
    if not d3_cache.exists():
        import urllib.request
        urllib.request.urlretrieve("https://d3js.org/d3.v7.min.js", d3_cache)
    d3_js = d3_cache.read_text()

    # Build legend from types actually present in the data
    used_node_types = sorted({n["type"] for n in nodes if n.get("type")})
    used_edge_types = sorted({e["type"] for e in edges})

    node_legend = "".join(
        f'<div class="legend-item"><span class="dot" style="background:{_color_for(t)}"></span>{t}</div>'
        for t in used_node_types
    )
    edge_legend = "".join(
        f'<div class="legend-item"><span class="dash" style="background:{_color_for(t, lightness=50)}"></span>{t.replace("_", " ").lower()}</div>'
        for t in used_edge_types
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>memagent — knowledge graph</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0f0f13; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; overflow: hidden; }}
  #canvas {{ width: 100vw; height: 100vh; }}

  .node circle {{ stroke-width: 2px; cursor: pointer; }}
  .node:hover circle {{ stroke: #fff; stroke-width: 2.5px; }}
  .node text {{ font-size: 12px; fill: #cbd5e1; pointer-events: none; text-shadow: 0 1px 4px #000c; }}

  .link {{ stroke-opacity: 0.7; fill: none; }}
  .link-label {{ font-size: 10px; fill: #64748b; pointer-events: none; }}

  #panel {{
    position: fixed; top: 0; right: 0;
    width: 320px; height: 100vh;
    background: #1e1e2e; border-left: 1px solid #2d2d3f;
    padding: 24px; overflow-y: auto;
    transform: translateX(100%); transition: transform 0.2s ease;
  }}
  #panel.open {{ transform: translateX(0); }}
  #panel h2 {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b; margin-bottom: 4px; }}
  #panel h1 {{ font-size: 20px; font-weight: 700; color: #f1f5f9; margin-bottom: 8px; }}
  #panel .type-badge {{
    display: inline-block; padding: 2px 8px; border-radius: 9999px;
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; margin-bottom: 16px;
  }}
  #panel .detail {{ font-size: 13px; color: #cbd5e1; line-height: 1.7; white-space: pre-wrap; }}
  #panel .relations {{ margin-top: 16px; }}
  #panel .rel-item {{ font-size: 12px; color: #94a3b8; padding: 3px 0; }}
  #panel .rel-type {{ font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-right: 6px; }}
  #panel .close {{ position: absolute; top: 16px; right: 16px; background: none; border: none; color: #64748b; font-size: 20px; cursor: pointer; }}
  #panel .close:hover {{ color: #f1f5f9; }}

  #legend {{
    position: fixed; bottom: 16px; left: 16px;
    background: #1e1e2edd; border: 1px solid #2d2d3f;
    border-radius: 8px; padding: 10px 14px;
    display: flex; flex-direction: column; gap: 6px;
    max-width: 220px;
  }}
  #legend .legend-section {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: #475569; margin-top: 4px; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 11px; color: #94a3b8; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  .dash {{ width: 16px; height: 2px; border-radius: 1px; flex-shrink: 0; }}

  #count {{ position: fixed; top: 16px; left: 16px; font-size: 11px; color: #475569; }}

  #edge-tooltip {{
    position: fixed; pointer-events: none;
    background: #1e1e2e; border: 1px solid #2d2d3f;
    border-radius: 4px; padding: 4px 8px;
    font-size: 11px; color: #94a3b8;
    display: none;
  }}
</style>
</head>
<body>
<svg id="canvas"></svg>

<div id="panel">
  <button class="close" onclick="closePanel()">✕</button>
  <h2 id="p-type-label">entity</h2>
  <h1 id="p-name"></h1>
  <span class="type-badge" id="p-badge"></span>
  <div class="detail" id="p-detail"></div>
  <div class="relations" id="p-relations"></div>
</div>

<div id="legend">
  <div class="legend-section">Nodes</div>
  {node_legend}
  <div class="legend-section">Edges</div>
  {edge_legend}
</div>
<div id="count"></div>
<div id="edge-tooltip"></div>

<script>{d3_js}</script>
<script>
const DATA = {graph_data};

// Build adjacency for panel relations
const adj = {{}};
DATA.edges.forEach(e => {{
  const s = typeof e.source === 'object' ? e.source.id : e.source;
  const t = typeof e.target === 'object' ? e.target.id : e.target;
  (adj[s] = adj[s] || []).push({{ id: t, type: e.type, dir: 'out' }});
  (adj[t] = adj[t] || []).push({{ id: s, type: e.type, dir: 'in' }});
}});
const nodeById = {{}};
DATA.nodes.forEach(n => nodeById[n.id] = n);

const svg = d3.select("#canvas");
const width = window.innerWidth, height = window.innerHeight;
svg.attr("viewBox", [0, 0, width, height]);
const g = svg.append("g");

svg.call(d3.zoom().scaleExtent([0.2, 5]).on("zoom", e => g.attr("transform", e.transform)));

const simulation = d3.forceSimulation(DATA.nodes)
  .force("link", d3.forceLink(DATA.edges).id(d => d.id).distance(150).strength(0.5))
  .force("charge", d3.forceManyBody().strength(-400))
  .force("center", d3.forceCenter(width / 2, height / 2))
  .force("collision", d3.forceCollide(44));

// Arrow markers per edge type
const defs = svg.append("defs");
const edgeTypes = [...new Set(DATA.edges.map(e => e.type))];
edgeTypes.forEach(type => {{
  const color = DATA.edges.find(e => e.type === type)?.color || "#334155";
  defs.append("marker")
    .attr("id", "arrow-" + type)
    .attr("viewBox", "0 -4 8 8")
    .attr("refX", 26).attr("refY", 0)
    .attr("markerWidth", 6).attr("markerHeight", 6)
    .attr("orient", "auto")
    .append("path").attr("d", "M0,-4L8,0L0,4").attr("fill", color).attr("opacity", 0.8);
}});

const tooltip = document.getElementById("edge-tooltip");

const link = g.append("g")
  .selectAll("line")
  .data(DATA.edges)
  .join("line")
  .attr("class", "link")
  .attr("stroke", d => d.color)
  .attr("stroke-width", 1.5)
  .attr("stroke-dasharray", d => d.dashed ? "4,3" : null)
  .attr("marker-end", d => `url(#arrow-${{d.type}})`)
  .on("mousemove", (e, d) => {{
    tooltip.style.display = "block";
    tooltip.style.left = (e.clientX + 12) + "px";
    tooltip.style.top = (e.clientY - 8) + "px";
    tooltip.textContent = d.type.replace(/_/g, " ").toLowerCase();
  }})
  .on("mouseleave", () => tooltip.style.display = "none");

const node = g.append("g")
  .selectAll("g")
  .data(DATA.nodes)
  .join("g")
  .attr("class", "node")
  .call(d3.drag()
    .on("start", (e, d) => {{ if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
    .on("drag",  (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
    .on("end",   (e, d) => {{ if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}))
  .on("click", (e, d) => openPanel(d));

// Entity nodes — circles
node.filter(d => d.shape === "circle").append("circle")
  .attr("r", 20)
  .attr("fill", d => d.color + "22")
  .attr("stroke", d => d.color)
  .attr("stroke-width", 2);

// Episode nodes — small squares
node.filter(d => d.shape === "square").append("rect")
  .attr("x", -10).attr("y", -10).attr("width", 20).attr("height", 20)
  .attr("fill", d => d.color + "33")
  .attr("stroke", d => d.color)
  .attr("stroke-width", 1.5);

node.append("text")
  .attr("dy", d => d.shape === "square" ? 24 : 34)
  .attr("text-anchor", "middle")
  .style("font-size", d => d.shape === "square" ? "10px" : "12px")
  .style("fill", d => d.shape === "square" ? "#64748b" : "#cbd5e1")
  .text(d => d.label);

simulation.on("tick", () => {{
  link
    .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
}});

document.getElementById("count").textContent =
  `${{DATA.nodes.length}} nodes · ${{DATA.edges.length}} edges`;

function openPanel(d) {{
  document.getElementById("p-type-label").textContent = d.type;
  document.getElementById("p-name").textContent = d.label;
  const badge = document.getElementById("p-badge");
  badge.textContent = d.type;
  badge.style.background = d.color + "33";
  badge.style.color = d.color;
  document.getElementById("p-detail").textContent = d.detail;

  // Relations
  const rels = adj[d.id] || [];
  const relDiv = document.getElementById("p-relations");
  if (rels.length) {{
    relDiv.innerHTML = "<h2 style='margin-top:16px;margin-bottom:8px'>Relations</h2>" +
      rels.map(r => {{
        const other = nodeById[r.id];
        const label = r.dir === 'out'
          ? `→ <span style="color:#f1f5f9">${{other?.label}}</span>`
          : `← <span style="color:#f1f5f9">${{other?.label}}</span>`;
        return `<div class="rel-item"><span class="rel-type" style="color:${{
          DATA.edges.find(e => {{
            const s = typeof e.source==='object'?e.source.id:e.source;
            const t = typeof e.target==='object'?e.target.id:e.target;
            return (s===d.id&&t===r.id)||(t===d.id&&s===r.id);
          }})?.color || '#94a3b8'
        }}">${{r.type.replace(/_/g,' ')}}</span>${{label}}</div>`;
      }}).join('');
  }} else {{
    relDiv.innerHTML = "";
  }}

  document.getElementById("panel").classList.add("open");
}}

function closePanel() {{
  document.getElementById("panel").classList.remove("open");
}}
</script>
</body>
</html>"""

    output_path.write_text(html)
