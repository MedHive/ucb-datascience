
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import dotenv_values
from neo4j import GraphDatabase
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from sklearn.metrics.pairwise import cosine_similarity

def get_driver():
    cfg = dotenv_values(".env")
    uri = cfg["NEO4J_URI"]
    user = cfg.get("NEO4J_USER", "neo4j")
    pw = cfg["NEO4J_PASSWORD"]
    return GraphDatabase.driver(uri, auth=(user, pw))

def fetch_triples_for_knum(driver, knum):
    triples = []
    entities = set()
    cypher = (
        "MATCH (s:Submission {hasKNumber: $k})-[*1..3]-(n) "
        "WITH DISTINCT s, n "
        "MATCH (s)-[r]->(n) "
        "RETURN startNode(r) AS s_node, type(r) AS pred, endNode(r) AS o_node"
    )
    with driver.session() as session:
        result = session.run(cypher, {"k": knum})
        for row in result:
            s_node = row["s_node"]
            o_node = row["o_node"]
            pred = row["pred"]
            s_uri = s_node.get("uri", f"node_{s_node.id}")
            o_uri = o_node.get("uri", f"node_{o_node.id}")
            triples.append((s_uri, pred, o_uri))
            entities.add(s_uri)
            entities.add(o_uri)
    return triples, entities

def run_kge_merge(k1, k2, out_csv, sim_threshold):
    driver = get_driver()
    kg1_triples, kg1_entities = fetch_triples_for_knum(driver, k1)
    kg2_triples, kg2_entities = fetch_triples_for_knum(driver, k2)
    driver.close()

    print(f"KG1 {k1} triples:", len(kg1_triples), "entities:", len(kg1_entities))
    print(f"KG2 {k2} triples:", len(kg2_triples), "entities:", len(kg2_entities))

    all_triples = np.array(kg1_triples + kg2_triples, dtype=str)
    tf = TriplesFactory.from_labeled_triples(all_triples)
    print("Total triples for training:", tf.num_triples)

    result = pipeline(
        training=tf,
        testing=tf,
        model="TransE",
        model_kwargs=dict(embedding_dim=64),
        training_kwargs=dict(num_epochs=200),
        device="cpu",
    )

    entity_embeddings = result.model.entity_representations[0]
    entity_to_id = result.training.entity_to_id
    all_entities = list(entity_to_id.keys())
    all_vectors = entity_embeddings(indices=None).detach().cpu().numpy()

    kg1_ids = [entity_to_id[e] for e in kg1_entities if e in entity_to_id]
    kg2_ids = [entity_to_id[e] for e in kg2_entities if e in entity_to_id]

    kg1_vecs = all_vectors[kg1_ids]
    kg2_vecs = all_vectors[kg2_ids]
    kg1_names = [all_entities[i] for i in kg1_ids]
    kg2_names = [all_entities[i] for i in kg2_ids]

    sim_matrix = cosine_similarity(kg2_vecs, kg1_vecs)

    rows = []
    for idx2, e2 in enumerate(kg2_names):
        sim_row = sim_matrix[idx2]
        idx1 = int(sim_row.argmax())
        score = float(sim_row[idx1])
        e1 = kg1_names[idx1]
        if score >= sim_threshold and e1 != e2:
            rows.append(
                dict(
                    k1=k1,
                    k2=k2,
                    entity1=e1,
                    entity2=e2,
                    similarity=score,
                )
            )

    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=["k1", "k2", "entity1", "entity2", "similarity"])
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("Candidates written to", out_path)
    print("Number of candidate SAMEAS pairs:", len(df))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k1", required=True)
    parser.add_argument("--k2", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    run_kge_merge(args.k1, args.k2, args.out, args.threshold)

if __name__ == "__main__":
    main()