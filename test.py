from neo4j import GraphDatabase
import os, certifi


os.environ["NEO4J_ENCRYPTION"] = "ENABLED"
os.environ["NEO4J_TRUST_ALL_CERTIFICATES"] = "TRUE"
os.environ["SSL_CERT_FILE"] = certifi.where()

uri = "neo4j+s://b873d406.databases.neo4j.io"
user = "neo4j"
password = "3vpmJ_ERN9xmWCG-iXHZGjdNxpfv5VJt2t20Ih3zqnA"

driver = GraphDatabase.driver(uri, auth=(user, password))
with driver.session() as session:
    result = session.run("RETURN 1 AS test")
    print(result.single())
