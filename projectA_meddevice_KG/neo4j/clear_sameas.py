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
        res = sess.run("MATCH ()-[r:SAMEAS]-() DELETE r RETURN count(r) AS c")
        print("Deleted SAMEAS relationships:", res.single()["c"])
    driver.close()

if __name__ == "__main__":
    main()
