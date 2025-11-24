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
        print("Distinct label pairs on SAMEAS:")
        res = sess.run(
            "MATCH (a)-[r:SAMEAS]-(b) "
            "RETURN DISTINCT labels(a) AS la, labels(b) AS lb"
        )
        for row in res:
            print("la:", row["la"], "lb:", row["lb"])

        print("\nSample SAMEAS pairs with uri:")
        res2 = sess.run(
            "MATCH (a)-[r:SAMEAS]-(b) "
            "RETURN labels(a) AS la, a.uri AS ua, "
            "labels(b) AS lb, b.uri AS ub "
            "LIMIT 20"
        )
        for row in res2:
            print("la:", row["la"], "ua:", row["ua"], "lb:", row["lb"], "ub:", row["ub"])
    driver.close()

if __name__ == "__main__":
    main()
