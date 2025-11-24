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
        q1 = """
        MATCH (n)
        WHERE any(k IN keys(n) WHERE toLower(toString(n[k])) CONTAINS 'rachel')
        RETURN labels(n) AS labels, n.uri AS uri, keys(n) AS props
        """
        rows = list(sess.run(q1))
        print("Nodes that mention rachel:", len(rows))
        for row in rows:
            print("labels:", row["labels"], "uri:", row.get("uri"), "props:", row["props"])

        q2 = """
        MATCH (n)
        WHERE any(k IN keys(n) WHERE toLower(toString(n[k])) CONTAINS 'rachel')
        OPTIONAL MATCH (n)-[r:SAMEAS]-(m)
        RETURN count(r) AS sameas_count
        """
        res2 = sess.run(q2)
        print("Total SAMEAS edges touching any rachel node:", res2.single()["sameas_count"])

    driver.close()

if __name__ == "__main__":
    main()
