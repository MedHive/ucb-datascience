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
        res = sess.run(
            "MATCH (n) "
            "RETURN labels(n) AS labels, count(n) AS c "
            "ORDER BY c DESC"
        )
        print("node labels and counts")
        for row in res:
            print(row["labels"], row["c"])
    driver.close()

if __name__ == "__main__":
    main()
