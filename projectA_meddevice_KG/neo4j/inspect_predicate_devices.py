import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

def main():
    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER", "neo4j")
    pw = os.getenv("NEO4J_PASSWORD")
    driver = GraphDatabase.driver(uri, auth=(user, pw))

    with driver.session() as sess:
        q = """
        MATCH (a:PredicateDevice)-[r:SAMEAS]-(b:PredicateDevice)
        RETURN DISTINCT a.uri AS ua, b.uri AS ub
        """
        rows = list(sess.run(q))
        print("PredicateDevice SAMEAS pairs:", len(rows))
        for row in rows:
            print("pair:", row["ua"], "<->", row["ub"])

        print("\nDetails for each PredicateDevice uri in SAMEAS:")
        q2 = """
        MATCH (n:PredicateDevice)
        WHERE n.uri IN $uris
        RETURN n.uri AS uri, labels(n) AS labels, properties(n) AS props
        ORDER BY uri
        """
        uris = sorted({r["ua"] for r in rows} | {r["ub"] for r in rows})
        if uris:
            res2 = sess.run(q2, uris=uris)
            for row in res2:
                print("uri:", row["uri"])
                print("  labels:", row["labels"])
                print("  props keys:", list(row["props"].keys()))
    driver.close()

if __name__ == "__main__":
    main()
