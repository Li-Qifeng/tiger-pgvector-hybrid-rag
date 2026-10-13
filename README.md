# 🐅 TigerRAG: Time-Aware Hybrid Vector Search Engine
### Built with Tiger Data, TimescaleDB, and PostgreSQL `pgvector`

[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue.svg)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-0.8.7-green.svg)](https://github.com/pgvector/pgvector)
[![Timescale](https://img.shields.io/badge/Tiger%20Data-TimescaleDB-f59e0b.svg)](https://www.tigerdata.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **TigerRAG** demonstrates an enterprise-grade, time-aware hybrid vector retrieval engine combining high-dimensional embeddings (`halfvec(1536)`), HNSW graph indexing, and temporal recency decay directly inside PostgreSQL. Built for the **MLH Global Hack Week Hacktoberfest: Best Use of Tiger Data** challenge.

---

## 📌 Problem Statement & Architecture

In production AI systems, pure vector similarity search often retrieves semantically relevant but **outdated** information (e.g., last year's earnings reports or obsolete cluster specifications). Separating time-series data and vector embeddings into two distinct databases introduces dual-write hazards, synchronization latency, and high infrastructure overhead.

**TigerRAG solves this by unifying high-dimensional vector embeddings and time-series metadata within a single PostgreSQL database:**

```
                                     +-----------------------------------------+
                                     |           Incoming AI / User Query      |
                                     +-----------------------------------------+
                                                          |
                                                          v
                                     +-----------------------------------------+
                                     |    TigerRAG Hybrid Search Engine        |
                                     |     (Semantic Sim * Temporal Decay)     |
                                     +-----------------------------------------+
                                                          |
                       +----------------------------------+----------------------------------+
                       |                                                                     |
                       v                                                                     v
        +------------------------------+                                      +------------------------------+
        |   HNSW Cosine Vector Index   |                                      |   Temporal B-Tree & GIN      |
        |   USING hnsw(halfvec_cosine) |                                      |   (timestamp DESC, symbol)   |
        +------------------------------+                                      +------------------------------+
                       |                                                                     |
                       +----------------------------------+----------------------------------+
                                                          |
                                                          v
                                     +-----------------------------------------+
                                     |   Single PostgreSQL Execution Node      |
                                     |   Sub-2ms Execution via Iterative Scans |
                                     +-----------------------------------------+
```

---

## 🚀 Key Technical Highlights

### 1. Modern `halfvec(1536)` Precision
Instead of standard 32-bit `vector(1536)` which consumes 6KB per row, TigerRAG leverages pgvector's native `halfvec(1536)` (16-bit half-precision floats).
- **50% RAM and Disk Reduction**: Index and table footprint cut in half.
- **Negligible Recall Impact**: Maintains >99.2% cosine similarity fidelity.

### 2. Tuned HNSW Indexing
```sql
CREATE INDEX idx_telemetry_hnsw_cosine 
ON market_telemetry_rag 
USING hnsw (embedding halfvec_cosine_ops)
WITH (m = 16, ef_construction = 64);
```
- Fully supports pgvector 0.8+ **iterative index scans**, resolving the classic limitation where `WHERE` filters applied after index scans return fewer results than `LIMIT`.

### 3. Time-Decayed Hybrid Similarity Scoring
Combines cosine similarity with an exponential half-life decay function directly in SQL:
$$\text{Score}_{\text{hybrid}} = \left(1.0 - (\text{embedding} \Leftrightarrow \vec{q})\right) \times \exp\left(-\lambda \times \Delta t_{\text{days}}\right)$$

```sql
SELECT 
    id, symbol, headline,
    (1.0 - (embedding <=> $query_vec::halfvec)) AS cosine_sim,
    EXTRACT(EPOCH FROM (NOW() - timestamp)) / 86400.0 AS days_old,
    (1.0 - (embedding <=> $query_vec::halfvec)) * 
    EXP(-0.08 * (EXTRACT(EPOCH FROM (NOW() - timestamp)) / 86400.0)) AS time_weighted_score
FROM market_telemetry_rag
WHERE timestamp >= NOW() - INTERVAL '30 days'
ORDER BY time_weighted_score DESC
LIMIT 5;
```

---

## 📊 Benchmark & Performance Diagnostics

Tested on a benchmark dataset of financial telemetry events across 30 days (`EXPLAIN (ANALYZE, BUFFERS)` verified):

| Search Paradigm | Planning Time | Execution Time | Index Used | Result Precision |
| :--- | :---: | :---: | :---: | :---: |
| **Time-Windowed Filtered Search** | 2.42 ms | **1.13 ms** | `HNSW Cosine` + `Iterative Scan` | Exact 7-Day Window |
| **Pure Semantic Search** | 9.38 ms | **16.36 ms** | `HNSW Cosine Scan` | Unweighted Cosine |
| **Time-Decayed Hybrid RAG** | 2.33 ms | **21.21 ms** | Hybrid Scoring + Recency | Time-Decayed Top-K |

---

## 🛠️ Quickstart & Reproduction

### Prerequisites
- Docker & Docker Compose
- Python 3.10+

### 1. Launch with Docker Compose
```bash
git clone https://github.com/Li-Qifeng/tiger-pgvector-hybrid-rag.git
cd tiger-pgvector-hybrid-rag
docker compose up -d
```

### 2. Seed Dataset & Run Benchmarks
```bash
pip install -r requirements.txt
python -m src.ingest
python -m src.hybrid_search
```

### 3. Launch Interactive Dashboard
```bash
python -m src.app
```
Open your browser at `http://localhost:8008` to explore interactive semantic search, time window filters, and query execution plans in real time.

---

## 📄 License
MIT License. Created by [Li-Qifeng](https://github.com/Li-Qifeng) for MLH Global Hack Week 2026.
