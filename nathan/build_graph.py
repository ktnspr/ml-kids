"""
Baut einen Beziehungsgraphen aus dem EPUB von 'Nathan der Weise'.

Schritte:
1. Text aus dem EPUB extrahieren (zipfile + html.parser).
2. In Chunks aufteilen.
3. Pro Chunk: Claude (Opus 4.7) extrahiert Figuren + Beziehungen aus DEM Text.
4. Alle Teilextraktionen werden konsolidiert + Steckbriefe formuliert.
5. HTML mit interaktivem Graph (vis.js) und Hover-Steckbrief schreiben.

Voraussetzung: ANTHROPIC_API_KEY in .env, plus:
    pip install anthropic python-dotenv
"""

from __future__ import annotations

import json
import os
import re
import sys
import zipfile
from html.parser import HTMLParser
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
EPUB_PATH = ROOT / "pg9186-images.epub"
OUT_HTML = ROOT / "nathan_graph.html"
OUT_JSON = ROOT / "graph_data.json"

MODEL = "claude-opus-4-7"
CHUNK_CHAR_SIZE = 18000


# --------------------------------------------------------------------------
# 1. EPUB-Text extrahieren
# --------------------------------------------------------------------------
class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._buf: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        if tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "li"):
            self._buf.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        if tag in ("p", "div", "h1", "h2", "h3", "h4", "li"):
            self._buf.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._buf.append(data)

    def text(self) -> str:
        return "".join(self._buf)


def extract_epub_text(path: Path) -> str:
    parts: list[str] = []
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))]
        names.sort()
        for name in names:
            with z.open(name) as f:
                raw = f.read().decode("utf-8", errors="ignore")
            p = _TextExtractor()
            p.feed(raw)
            parts.append(p.text())
    text = "\n\n".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int = CHUNK_CHAR_SIZE) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in text.split("\n\n"):
        plen = len(para) + 2
        if current_len + plen > size and current:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += plen
    if current:
        chunks.append("\n\n".join(current))
    return chunks


# --------------------------------------------------------------------------
# 2. Claude-Aufrufe
# --------------------------------------------------------------------------
EXTRACT_SYSTEM = """Du analysierst Lessings Drama 'Nathan der Weise'. Du bekommst einen Textausschnitt und extrahierst daraus Figuren, deren Entwicklung und Beziehungen.

Antworte AUSSCHLIESSLICH mit gültigem JSON in genau diesem Schema:
{
  "act_hint": "Aufzug/Szene falls erkennbar (z.B. 'I,1' oder 'III,7') oder null",
  "characters": [
    {
      "name": "Kanonischer Figurenname",
      "observations": ["kurze Beobachtung aus dem Text", "..."],
      "events": ["was passiert dieser Figur in diesem Ausschnitt - knapp, chronologisch", "..."]
    }
  ],
  "relations": [
    {
      "from": "Figurenname",
      "to": "Figurenname",
      "type": "kurze Beschreibung (z.B. 'Pflegevater', 'rettet aus Feuer', 'Bruder')",
      "evidence": "kurzes Zitat oder Paraphrase aus dem Text, das die Beziehung belegt"
    }
  ]
}

Regeln:
- Verwende immer den kanonischen Namen (Nathan, Recha, Daja, Tempelherr, Saladin, Sittah, Al-Hafi, Patriarch, Klosterbruder, Assad, ...).
- Nenne nur Figuren/Beziehungen, die im VORLIEGENDEN Ausschnitt vorkommen oder explizit erwähnt werden.
- 'events' = was tut/erlebt die Figur HIER (nicht allgemeine Eigenschaften).
- Wenn nichts Relevantes vorkommt: leere Listen zurückgeben.
- Kein Markdown, keine Kommentare, kein Vorwort - nur das JSON."""


MERGE_SYSTEM = """Du bekommst mehrere JSON-Teilextraktionen aus Lessings 'Nathan der Weise' (in chronologischer Reihenfolge der Chunks). Konsolidiere sie zu einem finalen Beziehungsgraphen mit Charakter-Timelines und erläuterten Beziehungen.

Antworte AUSSCHLIESSLICH mit gültigem JSON in diesem Schema:
{
  "characters": [
    {
      "id": "kurze_id_kleinbuchstaben",
      "name": "Anzeigename",
      "religion": "jude" | "christ" | "muslim" | "unbekannt",
      "role": "kurze Rollenbezeichnung (1 Zeile)",
      "profile": "Steckbrief in 3-5 Sätzen, basierend auf den Beobachtungen.",
      "timeline": [
        {
          "act": "Aufzug-Hinweis falls aus dem Hint ableitbar, sonst kurzer Phasen-Tag wie 'Anfang', 'Mitte', 'Ende'",
          "event": "ein konkretes Ereignis/Entwicklungsschritt der Figur, max 1 Satz"
        }
      ]
    }
  ],
  "relations": [
    {
      "from": "id",
      "to": "id",
      "label": "sehr kurze Bezeichnung (max 3 Worte) - erscheint am Pfeil",
      "kind": "familie" | "liebe" | "freundschaft" | "konflikt" | "dienst" | "religion" | "sonstige",
      "description": "1-2 Sätze: was diese Beziehung im Stück bedeutet, ihre Dynamik / Entwicklung."
    }
  ]
}

Regeln:
- Dedupliziere Figuren und Beziehungen.
- IDs in snake_case (z.B. 'nathan', 'tempelherr', 'al_hafi').
- 'from'/'to' verweisen auf die IDs aus 'characters'.
- Timeline: 3-6 Eintraege pro Hauptfigur, chronologisch geordnet (in der Reihenfolge der Chunks). Nebenfiguren duerfen weniger haben.
- Description: erklaert die Beziehung kurz fuer einen Leser, der das Stueck nicht kennt.
- Nimm nur Hauptfiguren und wichtige Nebenfiguren auf.
- Kein Markdown, kein Vorwort, kein Nachwort - nur das JSON."""


def call_claude_extract(client: Anthropic, chunk: str, idx: int, total: int) -> dict:
    print(f"  Chunk {idx}/{total} an Claude...", flush=True)
    message = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=[
            {
                "type": "text",
                "text": EXTRACT_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": f"Textausschnitt:\n\n{chunk}"}],
    )
    text = "".join(b.text for b in message.content if b.type == "text").strip()
    return _parse_json(text)


def call_claude_merge(client: Anthropic, partials: list[dict]) -> dict:
    print("  Konsolidierung an Claude...", flush=True)
    payload = json.dumps({"partials": partials}, ensure_ascii=False)
    message = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=MERGE_SYSTEM,
        messages=[{"role": "user", "content": payload}],
    )
    text = "".join(b.text for b in message.content if b.type == "text").strip()
    return _parse_json(text)


def _parse_json(text: str) -> dict:
    # Falls das Modell doch mal Markdown-Fences mitschickt, abstreifen
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  JSON-Parse-Fehler: {e}", file=sys.stderr)
        print(text[:500], file=sys.stderr)
        return {"characters": [], "relations": []}


# --------------------------------------------------------------------------
# 3. HTML rendern
# --------------------------------------------------------------------------
RELIGION_COLOR = {
    "jude": "#3498db",
    "christ": "#c0392b",
    "muslim": "#27ae60",
    "unbekannt": "#7f8c8d",
}

KIND_COLOR = {
    "familie": "#7f8c8d",
    "liebe": "#e91e63",
    "freundschaft": "#2980b9",
    "konflikt": "#c0392b",
    "dienst": "#16a085",
    "religion": "#8e44ad",
    "sonstige": "#95a5a6",
}


def build_html(data: dict) -> str:
    nodes = []
    for c in data.get("characters", []):
        color = RELIGION_COLOR.get(c.get("religion", "unbekannt"), "#7f8c8d")
        nodes.append(
            {
                "id": c["id"],
                "label": c["name"],
                "color": color,
                "religion": c.get("religion", "unbekannt"),
                "role": c.get("role", ""),
                "profile": c.get("profile", ""),
                "timeline": c.get("timeline", []),
            }
        )

    edges = []
    for i, r in enumerate(data.get("relations", [])):
        edges.append(
            {
                "id": f"e{i}",
                "from": r["from"],
                "to": r["to"],
                "label": r.get("label", ""),
                "color": KIND_COLOR.get(r.get("kind", "sonstige"), "#95a5a6"),
                "kind": r.get("kind", "sonstige"),
                "description": r.get("description", ""),
            }
        )

    payload = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>Nathan der Weise - Beziehungsgraph (aus EPUB)</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; font-family: Georgia, serif; background: #f5f1e8; color: #222; }}
  #header {{ padding: 12px 20px; background: #2c3e50; color: #fff; }}
  #header h1 {{ margin: 0; font-size: 20px; }}
  #header p {{ margin: 4px 0 0; font-size: 13px; opacity: 0.85; }}
  #network {{ width: 100%; height: calc(100vh - 70px); }}
  .legend {{
    position: absolute; top: 90px; right: 20px;
    background: rgba(255,255,255,0.95); padding: 10px 14px;
    border: 1px solid #ccc; border-radius: 6px; font-size: 12px;
    z-index: 5;
  }}
  .legend div {{ margin: 3px 0; }}
  .swatch {{ display: inline-block; width: 14px; height: 14px; vertical-align: middle; margin-right: 6px; border-radius: 50%; }}
  #tooltip {{
    position: absolute;
    display: none;
    max-width: 380px;
    background: #fffdf7;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 12px 14px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.22);
    font-size: 13px;
    line-height: 1.45;
    pointer-events: none;
    z-index: 10;
    color: #1a1a1a;
  }}
  #tooltip h3 {{ margin: 0 0 4px; font-size: 16px; color: #1a1a1a; }}
  #tooltip .role {{ font-style: italic; color: #555; margin-bottom: 6px; }}
  #tooltip .religion {{ display: inline-block; padding: 2px 8px; border-radius: 10px; color: #fff; font-size: 11px; margin-bottom: 8px; font-style: normal; }}
  #tooltip .section-label {{ margin-top: 10px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: #777; }}
  #tooltip .timeline {{ margin: 6px 0 0; padding: 0; list-style: none; border-left: 2px solid #c9b88a; }}
  #tooltip .timeline li {{ position: relative; padding: 4px 0 4px 14px; font-size: 12.5px; }}
  #tooltip .timeline li::before {{ content: ''; position: absolute; left: -5px; top: 10px; width: 8px; height: 8px; background: #c9b88a; border-radius: 50%; }}
  #tooltip .timeline .act {{ font-weight: bold; color: #6a4f1a; margin-right: 6px; }}
  #tooltip .edge-kind {{ display: inline-block; padding: 2px 8px; border-radius: 10px; color: #fff; font-size: 11px; margin-bottom: 6px; }}
</style>
</head>
<body>
<div id="header">
  <h1>Nathan der Weise - Beziehungen der Figuren</h1>
  <p>Extrahiert per Claude API aus dem EPUB. Hover ueber eine Figur fuer den Steckbrief.</p>
</div>
<div class="legend">
  <div><span class="swatch" style="background:#3498db"></span>Jude</div>
  <div><span class="swatch" style="background:#27ae60"></span>Muslim</div>
  <div><span class="swatch" style="background:#c0392b"></span>Christ</div>
  <div><span class="swatch" style="background:#7f8c8d"></span>Unbekannt</div>
</div>
<div id="network"></div>
<div id="tooltip"></div>
<script>
const DATA = {payload};

const RELIGION_LABEL = {{
  "jude": "Jude", "christ": "Christ", "muslim": "Muslim", "unbekannt": "Unbekannt"
}};
const KIND_LABEL = {{
  "familie": "Familie", "liebe": "Liebe", "freundschaft": "Freundschaft",
  "konflikt": "Konflikt", "dienst": "Dienst", "religion": "Religion", "sonstige": "Beziehung"
}};
const KIND_COLOR = {{
  "familie": "#7f8c8d", "liebe": "#e91e63", "freundschaft": "#2980b9",
  "konflikt": "#c0392b", "dienst": "#16a085", "religion": "#8e44ad", "sonstige": "#95a5a6"
}};

const nodes = new vis.DataSet(DATA.nodes.map(n => ({{
  id: n.id, label: n.label, color: {{ background: n.color, border: '#222' }},
  font: {{ color: '#fff', size: 15, face: 'Georgia', strokeColor: '#000', strokeWidth: 2 }},
  _meta: n
}})));

const edges = new vis.DataSet(DATA.edges.map(e => ({{
  id: e.id, from: e.from, to: e.to, label: e.label,
  color: {{ color: e.color, highlight: e.color, hover: '#222' }},
  font: {{
    size: 12, face: 'Georgia', color: '#1a1a1a',
    strokeColor: '#ffffff', strokeWidth: 4,
    align: 'middle'
  }},
  width: 1.6,
  hoverWidth: 1.2,
  selectionWidth: 1.6,
  arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }},
  smooth: {{ type: 'continuous' }},
  _meta: e
}})));

const network = new vis.Network(
  document.getElementById('network'),
  {{ nodes, edges }},
  {{
    nodes: {{ shape: 'dot', size: 22, borderWidth: 2 }},
    physics: {{ barnesHut: {{ gravitationalConstant: -9000, springLength: 170 }}, stabilization: {{ iterations: 250 }} }},
    interaction: {{ hover: true, tooltipDelay: 0, hoverConnectedEdges: false }}
  }}
);

const tooltip = document.getElementById('tooltip');

function showNodeTip(meta) {{
  const tl = (meta.timeline || []).map(t =>
    `<li><span class="act">${{t.act || ''}}</span>${{t.event || ''}}</li>`
  ).join('');
  tooltip.innerHTML = `
    <h3>${{meta.label}}</h3>
    <div class="religion" style="background:${{meta.color}}">${{RELIGION_LABEL[meta.religion] || meta.religion}}</div>
    <div class="role">${{meta.role || ''}}</div>
    <div>${{meta.profile || ''}}</div>
    ${{tl ? `<div class="section-label">Entwicklung</div><ul class="timeline">${{tl}}</ul>` : ''}}
  `;
  tooltip.style.display = 'block';
}}

function showEdgeTip(meta) {{
  const fromName = (nodes.get(meta.from) || {{}}).label || meta.from;
  const toName = (nodes.get(meta.to) || {{}}).label || meta.to;
  const kindColor = KIND_COLOR[meta.kind] || '#95a5a6';
  tooltip.innerHTML = `
    <h3>${{fromName}} &rarr; ${{toName}}</h3>
    <div class="edge-kind" style="background:${{kindColor}}">${{KIND_LABEL[meta.kind] || meta.kind}}</div>
    <div class="role">${{meta.label || ''}}</div>
    <div>${{meta.description || ''}}</div>
  `;
  tooltip.style.display = 'block';
}}

network.on('hoverNode', params => showNodeTip(nodes.get(params.node)._meta));
network.on('blurNode', () => {{ tooltip.style.display = 'none'; }});
network.on('hoverEdge', params => {{
  const edge = edges.get(params.edge);
  if (edge && edge._meta) showEdgeTip(edge._meta);
}});
network.on('blurEdge', () => {{ tooltip.style.display = 'none'; }});

document.addEventListener('mousemove', e => {{
  if (tooltip.style.display === 'block') {{
    const x = Math.min(e.clientX + 16, window.innerWidth - tooltip.offsetWidth - 10);
    const y = Math.min(e.clientY + 16, window.innerHeight - tooltip.offsetHeight - 10);
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
  }}
}});
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# 4. Main
# --------------------------------------------------------------------------
def main() -> None:
    load_dotenv(ROOT / ".env")
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY nicht in .env gefunden.")
    if not EPUB_PATH.exists():
        sys.exit(f"EPUB nicht gefunden: {EPUB_PATH}")

    print(f"Lese EPUB: {EPUB_PATH.name}")
    text = extract_epub_text(EPUB_PATH)
    print(f"  {len(text):,} Zeichen extrahiert.")

    chunks = chunk_text(text)
    print(f"  In {len(chunks)} Chunks aufgeteilt (~{CHUNK_CHAR_SIZE} Zeichen).")

    client = Anthropic()

    print("Extrahiere Figuren/Beziehungen pro Chunk:")
    partials: list[dict] = []
    for i, ch in enumerate(chunks, 1):
        partials.append(call_claude_extract(client, ch, i, len(chunks)))

    print("Konsolidiere alle Teilextraktionen:")
    final = call_claude_merge(client, partials)

    OUT_JSON.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  JSON gespeichert: {OUT_JSON.name}")

    OUT_HTML.write_text(build_html(final), encoding="utf-8")
    print(f"Fertig! HTML: {OUT_HTML.name}")
    print(
        f"  {len(final.get('characters', []))} Figuren, "
        f"{len(final.get('relations', []))} Beziehungen."
    )


if __name__ == "__main__":
    main()
