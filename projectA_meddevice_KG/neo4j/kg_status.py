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
        print("relationship counts")
        res_rel = sess.run("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c ORDER BY c DESC")
        for row in res_rel:
            print(f"{row['t']}: {row['c']}")

        print("\nnode counts by label")
        res_lab = sess.run("CALL db.labels() YIELD label "
                           "CALL { WITH label "
                           "  MATCH (n:`%s`) "
                           "  RETURN count(n) AS c "
                           "} "
                           "RETURN label, c ORDER BY c DESC" % "${label}")
        for row in res_lab:
            print(f"{row['label']}: {row['c']}")

        print("\nSAMEAS relationship count")
        res_same = sess.run("MATCH ()-[r:SAMEAS]-() RETURN count(r) AS c")
        print("SAMEAS:", res_same.single()["c"])

    driver.close()

if __name__ == "__main__":
    main()
