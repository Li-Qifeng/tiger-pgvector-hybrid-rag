"""
Data Ingestion Engine for Tiger Data x pgvector Hybrid RAG
Generates realistic financial news and market telemetry events with 1536-dimensional embeddings.
"""

import datetime
import json
import math
import random
import numpy as np
import psycopg2
from psycopg2.extras import execute_values

DB_URL = "host=localhost port=54329 user=postgres password=tigerpass dbname=postgres"

# Semantic clusters for realistic embeddings
TOPICS = {
    "AI_CHIPS": {
        "symbols": ["NVDA", "MSFT", "GOOGL"],
        "headlines": [
            "Next-gen GPU cluster delivers 3x throughput in AI training workloads",
            "Data center power constraints emerge as primary bottleneck for AI infrastructure",
            "High-bandwidth memory supply tightens amid soaring accelerator demand",
            "Hyperscalers increase multi-billion capex forecast for proprietary silicon",
            "Liquid cooling deployment accelerates across Tier-4 AI compute clusters"
        ],
        "sentiment": "Bullish",
        "impact_range": (0.80, 0.98),
        "cluster_bias": 0
    },
    "FED_MACRO": {
        "symbols": ["SPY", "QQQ", "BTC"],
        "headlines": [
            "Federal Reserve signals potential rate pause following softer inflation print",
            "Treasury yields retreat from multi-month highs as labor market cools",
            "Core PCE price index aligns with consensus expectations at 2.6%",
            "FOMC meeting minutes reveal cautious stance on monetary policy easing",
            "Global central banks coordinate liquidity facilities amid sovereign debt auctions"
        ],
        "sentiment": "Neutral",
        "impact_range": (0.50, 0.85),
        "cluster_bias": 1
    },
    "EV_SUPPLY": {
        "symbols": ["TSLA", "RIVN", "BYD"],
        "headlines": [
            "Next-generation lithium battery cell achieves 20% cost reduction milestone",
            "Autonomous driving fleet logs over 100 million unsupervised simulation miles",
            "Global EV export quotas shift as tariffs reshape North American supply chains",
            "Raw material futures stabilize following new mineral refinement agreement",
            "Supercharger network adoption broadens across legacy automotive OEMs"
        ],
        "sentiment": "Mixed",
        "impact_range": (0.40, 0.75),
        "cluster_bias": 2
    },
    "CLOUD_SAAS": {
        "symbols": ["MSFT", "AMZN", "CRM"],
        "headlines": [
            "Enterprise annual recurring revenue growth rebounds on GenAI workflow integrations",
            "Cloud infrastructure margin expansion beats quarterly analyst expectations",
            "Multi-tenant zero-trust architecture gains regulatory certification in EU",
            "Database workload migration trends favor PostgreSQL-compatible distributed engines",
            "Subscription renewal rates remain resilient amidst enterprise software consolidation"
        ],
        "sentiment": "Bullish",
        "impact_range": (0.60, 0.90),
        "cluster_bias": 3
    }
}

def generate_embedding(cluster_bias, dim=1536):
    """Generates a normalized 1536-dimensional vector clustered around topic direction."""
    vec = np.random.normal(0, 0.05, dim)
    # Inject cluster signature in specific dimension bands
    band_size = dim // 4
    start = cluster_bias * band_size
    vec[start:start + band_size] += np.random.uniform(0.3, 0.8, band_size)
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist()

def seed_data(total_records=300):
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    now = datetime.datetime.now(datetime.timezone.utc)
    records = []

    print(f"Generating {total_records} financial telemetry records...")
    for i in range(total_records):
        topic_key = random.choice(list(TOPICS.keys()))
        topic = TOPICS[topic_key]
        symbol = random.choice(topic["symbols"])
        headline = random.choice(topic["headlines"])
        summary = f"{headline}. Source reports robust trading volume and heightened options flow across {symbol} contracts."
        sentiment = topic["sentiment"] if random.random() > 0.2 else random.choice(["Bullish", "Bearish", "Neutral"])
        impact = round(random.uniform(*topic["impact_range"]), 2)

        # Spread timestamps across the last 30 days
        days_ago = random.uniform(0, 30)
        timestamp = now - datetime.timedelta(days=days_ago)

        metadata = {
            "topic": topic_key,
            "region": random.choice(["US", "APAC", "EMEA"]),
            "urgency": random.choice(["HIGH", "MEDIUM", "LOW"]),
            "analyst_rating": random.choice(["OVERWEIGHT", "NEUTRAL", "BUY"]),
            "word_count": random.randint(150, 600)
        }

        embedding = generate_embedding(topic["cluster_bias"])
        records.append((
            timestamp,
            symbol,
            f"TigerWire-{metadata['region']}",
            headline,
            summary,
            sentiment,
            impact,
            json.dumps(metadata),
            embedding
        ))

    insert_query = """
        INSERT INTO market_telemetry_rag (
            timestamp, symbol, source, headline, summary, sentiment, impact_score, metadata, embedding
        ) VALUES %s
    """

    print("Inserting records into PostgreSQL (halfvec)...")
    execute_values(cur, insert_query, records, template="(%s, %s, %s, %s, %s, %s, %s, %s, %s::halfvec)")
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM market_telemetry_rag;")
    count = cur.fetchone()[0]
    print(f"Ingestion complete! Total rows in market_telemetry_rag: {count}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    seed_data(350)
