import os, argparse, re, pandas as pd
from datetime import datetime
from neo4j import GraphDatabase
from dotenv import load_dotenv

def norm_label(lbl):
    m = {'Device':'MedicalDevice','Predicate':'PredicateDevice','PredicateDevice':'PredicateDevice','MedicalDevice':'MedicalDevice','Submission':'Submission','Applicant':'Applicant'}
    return m.get(lbl, lbl)

def parse_token(tok):
    if tok is None: return None, None
    s = str(tok)
    if ':' not in s: return None, s
    a,b = s.split(':',1)
    return norm_label(a.strip()), b.strip()

def is_node_token(tok):
    if tok is None:
        return False
    s = str(tok)
    if ':' not in s:
        return False
    return (
        s.startswith('Submission:')
        or s.startswith('Device:')
        or s.startswith('Predicate:')
        or s.startswith('Applicant:')
    )


def rel_type_from_pred(pred):
    return re.sub(r'[^A-Za-z0-9]','_', str(pred)).upper()

def create_constraints(tx):
    for lbl in ['Submission','MedicalDevice','PredicateDevice','Applicant']:
        tx.run(f'CREATE CONSTRAINT IF NOT EXISTS FOR (n:{lbl}) REQUIRE n.uri IS UNIQUE')

def merge_node(tx, label, uri):
    tx.run(f'MERGE (n:{label} {{uri:$u}})', u=uri)

def merge_rel(tx, s_label, s_uri, rtype, o_label, o_uri, props):
    q = f'''
    MATCH (s:{s_label} {{uri:$s}})
    MATCH (o:{o_label} {{uri:$o}})
    MERGE (s)-[r:{rtype}]->(o)
    SET r += $p
    '''
    tx.run(q, s=s_uri, o=o_uri, p=props)

def set_props(tx, label, uri, key, val, props):
    inc = {key: val}
    if props.get('source_documents') is not None:
        inc[key+'__source'] = props['source_documents']
    if props.get('intext_evidence') is not None:
        inc[key+'__evidence'] = props['intext_evidence']
    inc['last_updated_timestamp'] = props.get('last_updated_timestamp')
    tx.run(f'MATCH (n:{label} {{uri:$u}}) SET n += $m', u=uri, m=inc)

def main(csv_path):
    load_dotenv()
    uri = os.getenv('NEO4J_URI')
    user = os.getenv('NEO4J_USER','neo4j')
    pwd = os.getenv('NEO4J_PASSWORD')
    df = pd.read_csv(csv_path)
    driver = GraphDatabase.driver(uri, auth=(user,pwd))
    with driver.session() as sess:
        sess.execute_write(create_constraints)
        for _,row in df.iterrows():
            s_lbl, s_uri = parse_token(row['subject'] if 'subject' in row else None)
            pred = str(row['predicate']) if 'predicate' in row else None
            obj = row['object'] if 'object' in row else None
            props = {
                'source_documents': row.get('source_documents', None),
                'intext_evidence': row.get('intext_evidence', None),
                'creation_timestamp': row.get('creation_timestamp', None),
                'last_updated_timestamp': row.get('last_updated_timestamp', None) or datetime.utcnow().isoformat()
            }
            if not s_lbl or not s_uri or not pred: 
                continue
            sess.execute_write(merge_node, s_lbl, s_uri)
            if is_node_token(obj):
                o_lbl, o_uri = parse_token(obj)
                if o_lbl and o_uri:
                    sess.execute_write(merge_node, o_lbl, o_uri)
                    rtype = rel_type_from_pred(pred)
                    sess.execute_write(merge_rel, s_lbl, s_uri, rtype, o_lbl, o_uri, props)
            else:
                if obj is not None and str(obj) != '':
                    sess.execute_write(set_props, s_lbl, s_uri, pred, str(obj), props)
    driver.close()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    args = ap.parse_args()
    main(args.csv)
