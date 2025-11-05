import csv, os, re, sys, datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase

def camel_to_upper_snake(s):
    s = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', s)
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    return s.upper()

ALLOWED = {'Submission','Device','MedicalDevice','Predicate','PredicateDevice','Applicant'}

def is_node_token(s):
    m = re.match(r'^([A-Za-z]+):(.+)$', s.strip())
    return bool(m) and m.group(1) in ALLOWED

def label_from_token(s):
    p = re.match(r'^([A-Za-z]+):', s.strip()).group(1)
    return {'Device':'MedicalDevice','Predicate':'PredicateDevice'}.get(p, p)

def ensure_node(sess, label, uri, ct, ut):
    q = f"MERGE (n:{label} {{uri:$u}}) ON CREATE SET n.creation_timestamp=$ct SET n.last_updated_timestamp=$ut"
    sess.run(q, u=uri, ct=ct, ut=ut)

def main():
    load_dotenv("projectA_meddevice_KG/.env")
    uri=os.environ["NEO4J_URI"]; user=os.environ["NEO4J_USERNAME"]; pwd=os.environ["NEO4J_PASSWORD"]
    csv_path=sys.argv[1]
    drv=GraphDatabase.driver(uri,auth=(user,pwd))
    created=0
    with drv.session() as sess, open(csv_path, newline='') as f:
        r=csv.DictReader(f)
        for row in r:
            subj=row['subject'].strip()
            pred=row['predicate'].strip()
            obj=row['object'].strip()
            src=(row.get('source_documents') or '').strip()
            ev=(row.get('intext_evidence') or '').strip()
            ct=(row.get('creation_timestamp') or datetime.datetime.utcnow().isoformat())
            ut=(row.get('last_updated_timestamp') or ct)
            if not is_node_token(subj):
                continue
            s_label=label_from_token(subj)
            ensure_node(sess, s_label, subj, ct, ut)
            if is_node_token(obj):
                o_label=label_from_token(obj)
                ensure_node(sess, o_label, obj, ct, ut)
                rel=camel_to_upper_snake(pred)
                q=(f"MATCH (s:{s_label} {{uri:$suri}}), (o:{o_label} {{uri:$ouri}}) "
                   f"MERGE (s)-[r:{rel}]->(o) "
                   f"SET r.source_documents=$src, r.intext_evidence=$ev, "
                   f"    r.creation_timestamp=coalesce(r.creation_timestamp,$ct), r.last_updated_timestamp=$ut")
                sess.run(q, suri=subj, ouri=obj, src=src, ev=ev, ct=ct, ut=ut)
            else:
                prop=re.sub(r'[^A-Za-z0-9_]', '_', pred)
                q=(f"MATCH (s:{s_label} {{uri:$suri}}) "
                   f"SET s.`{prop}`=$val, s.`{prop}__source`=$src, s.`{prop}__evidence`=$ev, s.last_updated_timestamp=$ut")
                sess.run(q, suri=subj, val=obj, src=src, ev=ev, ut=ut)
            created+=1
    print(f"Imported/merged {created} triples into Neo4j.")
    drv.close()

if __name__ == "__main__":
    main()
