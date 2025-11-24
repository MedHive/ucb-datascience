import argparse
from datetime import datetime

import pandas as pd
from dotenv import dotenv_values
from neo4j import GraphDatabase


ALLOWED_LABELS = ["PredicateDevice", "Applicant"]


def get_driver():
    cfg = dotenv_values(".env")
    uri = cfg["NEO4J_URI"]
    user = cfg.get("NEO4J_USER", "neo4j")
    pw = cfg["NEO4J_PASSWORD"]
    return GraphDatabase.driver(uri, auth=(user, pw))


def merge_sameas(tx, uri1, uri2, similarity, timestamp):
    query = """
    MATCH (a {uri: $u1}), (b {uri: $u2})
    WHERE a <> b
    WITH a, b, labels(a) AS la, labels(b) AS lb
    WHERE la = lb
      AND any(l IN la WHERE l IN $allowed)
      AND (
            ("PredicateDevice" IN la AND coalesce(a.hasPredicateName, "") = coalesce(b.hasPredicateName, ""))
         OR ("Applicant" IN la AND coalesce(a.hasContactEmail, "") <> "" AND a.hasContactEmail = b.hasContactEmail)
      )
    MERGE (a)-[r:SAMEAS]-(b)
    ON CREATE SET
        r.similarity = $sim,
        r.created_at = $ts
    ON MATCH SET
        r.similarity = CASE
            WHEN $sim > coalesce(r.similarity, 0) THEN $sim
            ELSE r.similarity
        END,
        r.last_updated = $ts
    RETURN count(r) AS c
    """
    res = tx.run(
        query,
        u1=uri1,
        u2=uri2,
        sim=float(similarity),
        ts=timestamp,
        allowed=ALLOWED_LABELS,
    )
    record = res.single()
    return record["c"] if record is not None else 0


def apply_sameas(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["entity1", "entity2"])
    cfg = dotenv_values(".env")
    uri = cfg["NEO4J_URI"]
    user = cfg.get("NEO4J_USER", "neo4j")
    pw = cfg["NEO4J_PASSWORD"]
    driver = GraphDatabase.driver(uri, auth=(user, pw))
    ts = datetime.utcnow().isoformat()
    total = 0
    with driver.session() as session:
        for row in df.itertuples(index=False):
            total += session.execute_write(
                merge_sameas,
                getattr(row, "entity1"),
                getattr(row, "entity2"),
                getattr(row, "similarity"),
                ts,
            )
    driver.close()
    print("SAMEAS relationships processed:", total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    apply_sameas(args.csv)


if __name__ == "__main__":
    main()
