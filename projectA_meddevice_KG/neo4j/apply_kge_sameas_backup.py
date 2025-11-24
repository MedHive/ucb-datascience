import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import dotenv_values
from neo4j import GraphDatabase


def get_driver():
    cfg = dotenv_values(".env")
    uri = cfg["NEO4J_URI"]
    user = cfg.get("NEO4J_USER", "neo4j")
    pw = cfg["NEO4J_PASSWORD"]
    return GraphDatabase.driver(uri, auth=(user, pw))


def apply_sameas(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print("rows in CSV:", len(df))
    if df.empty:
        print("CSV is empty, no SAMEAS pairs to write")
        return

    required = ["k1", "k2", "entity1", "entity2", "similarity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("missing columns:", missing)
        return

    cfg = dotenv_values(".env")
    uri = cfg["NEO4J_URI"]
    user = cfg.get("NEO4J_USER", "neo4j")
    pw = cfg["NEO4J_PASSWORD"]
    driver = GraphDatabase.driver(uri, auth=(user, pw))

    ts = datetime.utcnow().isoformat()
    method = "pykeen_kge"
    source = csv_path.name

    count = 0
    with driver.session() as session:
        for row in df.itertuples(index=False):
            if row.entity1 == row.entity2:
                continue
            session.run(
                """
                MATCH (n1 {uri: $u1})
                MATCH (n2 {uri: $u2})
                MERGE (n1)-[r:SAMEAS]->(n2)
                ON CREATE SET
                  r.k1 = $k1,
                  r.k2 = $k2,
                  r.similarity = $sim,
                  r.method = $method,
                  r.source_documents = $source,
                  r.intext_evidence = $evidence,
                  r.creation_timestamp = $ts,
                  r.last_updated_timestamp = $ts
                ON MATCH SET
                  r.last_updated_timestamp = $ts,
                  r.similarity = CASE
                    WHEN $sim > coalesce(r.similarity, 0) THEN $sim
                    ELSE r.similarity
                  END
                """,
                u1=row.entity1,
                u2=row.entity2,
                k1=row.k1,
                k2=row.k2,
                sim=float(row.similarity),
                method=method,
                source=source,
                evidence=f"embedding similarity {row.similarity}",
                ts=ts,
            )
            count += 1

    driver.close()
    print("SAMEAS relationships processed:", count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    apply_sameas(args.csv)


if __name__ == "__main__":
    main()
