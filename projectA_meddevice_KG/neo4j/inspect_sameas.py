import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

def get_driver():
    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER", "neo4j")
    pw = os.getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(uri, auth=(user, pw))

def main():
    driver = get_driver()
    with driver.session() as sess:
        res1 = sess.run("MATCH (:Person)-[r:SAMEAS]-() RETURN count(r) AS c")
        print("Person SAMEAS relationships:", res1.single()["c"])
        res2 = sess.run("MATCH ()-[r:SAMEAS]-() RETURN count(r) AS c")
        print("Total SAMEAS relationships:", res2.single()["c"])
    driver.close()

if __name__ == "__main__":
    main()
