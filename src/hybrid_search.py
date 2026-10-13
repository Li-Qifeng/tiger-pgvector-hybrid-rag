"""
Hybrid Time-Series & Vector Search Engine
Demonstrates:
1. Pure Semantic Vector Search (HNSW Cosine Distance)
2. Filtered Vector Search (Temporal chunking & metadata)
3. Time-Decayed Hybrid Scoring (Semantic + Recency Exponential Decay)
4. EXPLAIN ANALYZE Performance Diagnostics
"""

import json
import time
import psycopg2
import numpy as np
from src.ingest import generate_embedding, DB_URL

def execute_query_with_explain(cur, query_name, sql, params):
    print(f"\n{'='*70}\n[RUNNING QUERY]: {query_name}\n{'='*70}")
    
    # Run EXPLAIN ANALYZE
    explain_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}"
    cur.execute(explain_sql, params)
    explain_plan = cur.fetchone()[0][0]
    exec_time = explain_plan.get("Execution Time")
    plan_time = explain_plan.get("Planning Time")
    
    # Run actual query
    start_t = time.perf_counter()
    cur.execute(sql, params)
    rows = cur.fetchall()
    client_elapsed_ms = (time.perf_counter() - start_t) * 1000

    print(f"Planning Time:   {plan_time:.3f} ms")
    print(f"Execution Time:  {exec_time:.3f} ms")
    print(f"Client Latency:  {client_elapsed_ms:.3f} ms")
    print(f"Results Returned: {len(rows)}")
    
    return rows, {
        "planning_time_ms": plan_time,
        "execution_time_ms": exec_time,
        "client_latency_ms": client_elapsed_ms,
        "plan_tree": explain_plan["Plan"]
    }

def run_hybrid_search_demonstration():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    # Query embedding: AI chip cluster power & supply chain (Cluster Bias 0)
    query_vec = generate_embedding(cluster_bias=0)

    # -------------------------------------------------------------
    # 1. Pure Semantic Search (HNSW Index Scan)
    # -------------------------------------------------------------
    sql_pure_vector = """
        SELECT 
            id, 
            timestamp, 
            symbol, 
            headline, 
            sentiment,
            impact_score,
            (1.0 - (embedding <=> %s::halfvec)) AS cosine_similarity
        FROM market_telemetry_rag
        ORDER BY embedding <=> %s::halfvec
        LIMIT 5;
    """
    rows_v, plan_v = execute_query_with_explain(
        cur, 
        "1. Pure Semantic Vector Search (HNSW)", 
        sql_pure_vector, 
        (query_vec, query_vec)
    )
    for r in rows_v:
        dt = r[1].strftime('%Y-%m-%d %H:%M')
        print(f"  [{r[2]}] (Sim: {r[6]:.4f} | {dt}) {r[3][:60]}...")

    # -------------------------------------------------------------
    # 2. Filtered Search (Recent 7 Days + AI Tickers)
    # -------------------------------------------------------------
    sql_filtered_vector = """
        SELECT 
            id, 
            timestamp, 
            symbol, 
            headline, 
            sentiment,
            impact_score,
            (1.0 - (embedding <=> %s::halfvec)) AS cosine_similarity
        FROM market_telemetry_rag
        WHERE timestamp >= NOW() - INTERVAL '7 days'
          AND symbol IN ('NVDA', 'MSFT', 'GOOGL')
        ORDER BY embedding <=> %s::halfvec
        LIMIT 5;
    """
    rows_f, plan_f = execute_query_with_explain(
        cur, 
        "2. Time-Windowed & Symbol Filtered Vector Search", 
        sql_filtered_vector, 
        (query_vec, query_vec)
    )
    for r in rows_f:
        dt = r[1].strftime('%Y-%m-%d %H:%M')
        print(f"  [{r[2]}] (Sim: {r[6]:.4f} | {dt}) {r[3][:60]}...")

    # -------------------------------------------------------------
    # 3. Time-Decayed Hybrid Similarity Search
    # Combines Semantic Cosine Score with Exponential Recency Decay:
    # HybridScore = CosineSim * exp(-0.08 * days_ago)
    # -------------------------------------------------------------
    sql_hybrid_decay = """
        SELECT 
            id, 
            timestamp, 
            symbol, 
            headline, 
            sentiment,
            impact_score,
            (1.0 - (embedding <=> %s::halfvec)) AS raw_similarity,
            EXTRACT(EPOCH FROM (NOW() - timestamp)) / 86400.0 AS days_old,
            (1.0 - (embedding <=> %s::halfvec)) * 
            EXP(-0.08 * (EXTRACT(EPOCH FROM (NOW() - timestamp)) / 86400.0)) AS time_weighted_score
        FROM market_telemetry_rag
        WHERE timestamp >= NOW() - INTERVAL '30 days'
        ORDER BY time_weighted_score DESC
        LIMIT 5;
    """
    rows_h, plan_h = execute_query_with_explain(
        cur, 
        "3. Time-Decayed Hybrid RAG Search (Semantic + Recency Weight)", 
        sql_hybrid_decay, 
        (query_vec, query_vec)
    )
    for r in rows_h:
        dt = r[1].strftime('%Y-%m-%d %H:%M')
        print(f"  [{r[2]}] (FinalScore: {r[8]:.4f} | RawSim: {r[6]:.4f} | Age: {r[7]:.1f}d) {r[3][:55]}...")

    cur.close()
    conn.close()

    # Save benchmark summary for README & Submission
    benchmarks = {
        "pure_vector_exec_ms": plan_v["execution_time_ms"],
        "filtered_vector_exec_ms": plan_f["execution_time_ms"],
        "hybrid_decay_exec_ms": plan_h["execution_time_ms"]
    }
    with open("/root/tiger-pgvector-hybrid-rag/benchmark_results.json", "w") as f:
        json.dump(benchmarks, f, indent=2)
    print("\nBenchmark results successfully logged!")

if __name__ == "__main__":
    run_hybrid_search_demonstration()
