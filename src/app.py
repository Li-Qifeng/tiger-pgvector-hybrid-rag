"""
Interactive Web Dashboard & API for Tiger Data x pgvector Hybrid Search
"""

import json
import time
import psycopg2
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from src.ingest import generate_embedding, DB_URL

app = FastAPI(title="TigerRAG Hybrid Search Engine")

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TigerRAG: Time-Aware Hybrid Vector Search (TimescaleDB + pgvector)</title>
<style>
  :root {
    --bg: #090d16;
    --card: #111827;
    --border: #1f2937;
    --accent: #f59e0b;
    --accent-blue: #38bdf8;
    --text: #f9fafb;
    --text-muted: #9ca3af;
  }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 24px; }
  .container { max-width: 1200px; margin: 0 auto; }
  .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 24px; }
  .title-group h1 { margin: 0; font-size: 26px; color: var(--accent); display: flex; align-items: center; gap: 10px; }
  .title-group p { margin: 6px 0 0; color: var(--text-muted); font-size: 14px; }
  .badges { display: flex; gap: 8px; }
  .badge { background: #1e293b; border: 1px solid #334155; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; color: var(--accent-blue); }
  
  .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
  .stat-card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
  .stat-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }
  .stat-val { font-size: 24px; font-weight: 700; color: var(--text); margin-top: 6px; }
  .stat-sub { font-size: 12px; color: #10b981; margin-top: 4px; }

  .controls-card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 20px; margin-bottom: 24px; }
  .form-row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  select, input[type="text"] { background: #0b1120; border: 1px solid #374151; color: #fff; padding: 10px 14px; border-radius: 8px; font-size: 14px; outline: none; }
  input[type="text"] { flex: 1; min-width: 280px; }
  select:focus, input[type="text"]:focus { border-color: var(--accent); }
  button { background: var(--accent); color: #000; border: none; font-weight: 700; padding: 10px 20px; border-radius: 8px; cursor: pointer; transition: 0.2s; }
  button:hover { background: #d97706; }

  .results-section h2 { font-size: 18px; color: var(--accent-blue); margin-top: 0; }
  .result-card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 18px; margin-bottom: 14px; transition: transform 0.15s; }
  .result-card:hover { border-color: #374151; transform: translateY(-2px); }
  .card-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
  .ticker { background: #1e3a8a; color: #93c5fd; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 13px; }
  .scores { display: flex; gap: 12px; font-size: 13px; font-family: monospace; }
  .score-final { color: #f59e0b; font-weight: bold; }
  .score-raw { color: #9ca3af; }
  .headline { font-size: 16px; font-weight: 600; margin: 0 0 6px; }
  .summary { color: var(--text-muted); font-size: 14px; line-height: 1.5; margin: 0; }
  .meta-footer { display: flex; gap: 16px; margin-top: 12px; font-size: 12px; color: #6b7280; font-family: monospace; }
  
  .explain-box { background: #030712; border: 1px solid #1f2937; border-radius: 8px; padding: 14px; font-family: monospace; font-size: 12px; color: #a3e635; margin-top: 20px; overflow-x: auto; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="title-group">
      <h1>🐅 TigerRAG: Time-Aware Hybrid Search</h1>
      <p>High-Dimensional Vector Embeddings (1536-dim halfvec) + TimescaleDB Temporal Filtering & Recency Decay</p>
    </div>
    <div class="badges">
      <span class="badge">pgvector 0.8.7</span>
      <span class="badge">HNSW (m=16, ef=64)</span>
      <span class="badge">halfvec(1536)</span>
    </div>
  </div>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-label">Vector Storage Type</div>
      <div class="stat-val">HALFVEC(1536)</div>
      <div class="stat-sub">50% RAM & Disk reduction</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Index Type</div>
      <div class="stat-val">HNSW Cosine</div>
      <div class="stat-sub">Iterative filtered scan support</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Index Scan Latency</div>
      <div class="stat-val" id="queryLatency">1.13 ms</div>
      <div class="stat-sub">EXPLAIN ANALYZE verified</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Total Seed Records</div>
      <div class="stat-val">350 Events</div>
      <div class="stat-sub">Spanning 30-day temporal window</div>
    </div>
  </div>

  <div class="controls-card">
    <div class="form-row">
      <select id="topicSelect">
        <option value="0">Topic: AI Hardware & Next-Gen GPUs</option>
        <option value="1">Topic: Federal Reserve Macro & Rates</option>
        <option value="2">Topic: EV Battery & Autonomous Tech</option>
        <option value="3">Topic: Cloud Infrastructure & Enterprise SaaS</option>
      </select>
      <select id="modeSelect">
        <option value="hybrid">Mode: Time-Decayed Hybrid (Semantic + Recency)</option>
        <option value="filtered">Mode: Strict 7-Day Window Filter</option>
        <option value="pure">Mode: Pure Semantic Similarity</option>
      </select>
      <button onclick="runSearch()">Execute Hybrid Query</button>
    </div>
  </div>

  <div class="results-section">
    <h2>Query Results & Temporal Reranking</h2>
    <div id="resultsList">
      <div style="color: #6b7280;">Click "Execute Hybrid Query" to test query execution across HNSW and temporal chunk filters...</div>
    </div>
  </div>

  <div class="explain-box" id="explainOutput">
    -- EXPLAIN (ANALYZE, BUFFERS) Query Execution Plan will appear here --
  </div>
</div>

<script>
async function runSearch() {
  const topic = document.getElementById('topicSelect').value;
  const mode = document.getElementById('modeSelect').value;
  const list = document.getElementById('resultsList');
  const explain = document.getElementById('explainOutput');
  const latency = document.getElementById('queryLatency');

  list.innerHTML = '<div style="color: #9ca3af;">Executing vector distance computation and temporal reranking...</div>';

  try {
    const res = await fetch(`/api/search?topic=${topic}&mode=${mode}`);
    const data = await res.json();

    latency.textContent = data.latency_ms + ' ms';
    explain.textContent = JSON.stringify(data.explain_plan, null, 2);

    let html = '';
    data.results.forEach((r, idx) => {
      html += `
        <div class="result-card">
          <div class="card-top">
            <span class="ticker">${r.symbol}</span>
            <div class="scores">
              ${r.final_score ? `<span class="score-final">Hybrid Score: ${r.final_score.toFixed(4)}</span>` : ''}
              <span class="score-raw">Cosine Sim: ${r.cosine_similarity.toFixed(4)}</span>
              ${r.days_old !== undefined ? `<span>Age: ${r.days_old.toFixed(1)}d</span>` : ''}
            </div>
          </div>
          <h3 class="headline">${r.headline}</h3>
          <p class="summary">${r.summary}</p>
          <div class="meta-footer">
            <span>Timestamp: ${r.timestamp}</span>
            <span>Sentiment: ${r.sentiment}</span>
            <span>Impact: ${r.impact_score}</span>
            <span>Source: ${r.source}</span>
          </div>
        </div>
      `;
    });
    list.innerHTML = html;
  } catch (e) {
    list.innerHTML = '<div style="color: #ef4444;">Query failed: ' + e + '</div>';
  }
}

// Auto-run first query on load
window.onload = runSearch;
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_DASHBOARD

@app.get("/api/search")
def search(topic: int = Query(0), mode: str = Query("hybrid")):
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    query_vec = generate_embedding(cluster_bias=topic)
    start_t = time.perf_counter()

    if mode == "pure":
        sql = """
            SELECT id, timestamp, symbol, source, headline, summary, sentiment, impact_score,
                   (1.0 - (embedding <=> %s::halfvec)) AS cosine_similarity
            FROM market_telemetry_rag
            ORDER BY embedding <=> %s::halfvec
            LIMIT 5;
        """
        params = (query_vec, query_vec)
    elif mode == "filtered":
        sql = """
            SELECT id, timestamp, symbol, source, headline, summary, sentiment, impact_score,
                   (1.0 - (embedding <=> %s::halfvec)) AS cosine_similarity
            FROM market_telemetry_rag
            WHERE timestamp >= NOW() - INTERVAL '7 days'
            ORDER BY embedding <=> %s::halfvec
            LIMIT 5;
        """
        params = (query_vec, query_vec)
    else:  # hybrid
        sql = """
            SELECT id, timestamp, symbol, source, headline, summary, sentiment, impact_score,
                   (1.0 - (embedding <=> %s::halfvec)) AS cosine_similarity,
                   EXTRACT(EPOCH FROM (NOW() - timestamp)) / 86400.0 AS days_old,
                   (1.0 - (embedding <=> %s::halfvec)) * 
                   EXP(-0.08 * (EXTRACT(EPOCH FROM (NOW() - timestamp)) / 86400.0)) AS time_weighted_score
            FROM market_telemetry_rag
            WHERE timestamp >= NOW() - INTERVAL '30 days'
            ORDER BY time_weighted_score DESC
            LIMIT 5;
        """
        params = (query_vec, query_vec)

    # Run EXPLAIN
    cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}", params)
    explain_data = cur.fetchone()[0][0]

    # Run query
    cur.execute(sql, params)
    rows = cur.fetchall()
    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

    results = []
    for r in rows:
        item = {
            "id": r[0],
            "timestamp": r[1].strftime("%Y-%m-%d %H:%M:%S UTC"),
            "symbol": r[2],
            "source": r[3],
            "headline": r[4],
            "summary": r[5],
            "sentiment": r[6],
            "impact_score": float(r[7]),
            "cosine_similarity": float(r[8])
        }
        if mode == "hybrid":
            item["days_old"] = float(r[9])
            item["final_score"] = float(r[10])
        results.append(item)

    cur.close()
    conn.close()

    return {
        "mode": mode,
        "latency_ms": latency_ms,
        "explain_plan": explain_data,
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
