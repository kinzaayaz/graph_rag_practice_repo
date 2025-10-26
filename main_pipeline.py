import os
from dotenv import load_dotenv
from typing import List
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os, certifi
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars
from pydantic import BaseModel, Field

load_dotenv()

os.environ["NEO4J_ENCRYPTION"] = "ENABLED"
os.environ["NEO4J_TRUST_ALL_CERTIFICATES"] = "TRUE"
os.environ["SSL_CERT_FILE"] = certifi.where()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

print("🔗 Trying to connect to:", NEO4J_URI)
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run("RETURN 1 AS test")
        print(result.single())
    print("✅ Connection successful (direct test)!")
except Exception as e:
    print("❌ Connection failed (direct test):", e)

# --- Step 2: Test LangChain Neo4jGraph connection ---
graph = None

try:
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
    print("✅ Connected successfully to Neo4jGraph!")
except Exception as e:
    print("❌ Failed to connect Neo4jGraph:", e)

llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", api_key=OPENAI_API_KEY)


class Entities(BaseModel):
    names: List[str] = Field(..., description="List of extracted entities from the question")

entity_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract all relevant named entities from the given text."),
    ("human", "Extract entities from this question: {question}")
])
entity_chain = entity_prompt | llm.with_structured_output(Entities, method="function_calling")

def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

def structured_retriever_all(question: str, exclude_rels=None, batch_size=5):
    if not graph:
        raise ValueError("Neo4j graph connection not initialized.")

    """
    Retrieve all relationships for entities related to a question in chunks,
    then combine results into a single output.

    Args:
        question (str): The query string.
        exclude_rels (list[str], optional): Relationship types to exclude. Default: ['MENTIONS'].
        batch_size (int, optional): Number of entities to process per batch.

    Returns:
        list: All relationships as tuples (source, rel, target)
    """
    exclude_rels = exclude_rels or ['MENTIONS']
    all_relationships = []

    # Get entities for the question
    entities = entity_chain.invoke({"question": question})
    entity_names = getattr(entities, "names", entities) if entities else []
    if isinstance(entity_names, str):
        entity_names = [entity_names]

    # Process entities in batches to avoid token limits
    for i in range(0, len(entity_names), batch_size):
        batch = entity_names[i:i+batch_size]
        batch_relationships = []

        for entity in batch:
            try:
                cypher = f"""
                CALL db.index.fulltext.queryNodes('entity', $query)
                YIELD node
                MATCH (node)-[r]->(neighbor)
                WHERE NOT type(r) IN $exclude_rels
                RETURN node.id AS source_id, type(r) AS rel_type, neighbor.id AS target_id
                UNION
                MATCH (neighbor)-[r]->(node)
                WHERE NOT type(r) IN $exclude_rels
                RETURN neighbor.id AS source_id, type(r) AS rel_type, node.id AS target_id
                """
                rows = graph.query(cypher, {"query": generate_full_text_query(entity), "exclude_rels": exclude_rels})
            except Exception:
                # Fallback to substring match
                cypher_fallback = f"""
                MATCH (n)
                WHERE toLower(n.id) CONTAINS toLower($q)
                   OR any(lbl IN labels(n) WHERE toLower(lbl) CONTAINS toLower($q))
                MATCH (n)-[r]->(neighbor)
                WHERE NOT type(r) IN $exclude_rels
                RETURN n.id AS source_id, type(r) AS rel_type, neighbor.id AS target_id
                UNION
                MATCH (neighbor)-[r]->(n)
                WHERE NOT type(r) IN $exclude_rels
                RETURN neighbor.id AS source_id, type(r) AS rel_type, n.id AS target_id
                """
                rows = graph.query(cypher_fallback, {"q": entity, "exclude_rels": exclude_rels})

            # Append results
            for row in rows:
                if isinstance(row, dict):
                    src = row.get("source_id")
                    rel = row.get("rel_type")
                    tgt = row.get("target_id")
                else:
                    try:
                        src, rel, tgt = row
                    except Exception:
                        continue
                if src and rel and tgt:
                    batch_relationships.append((src, rel, tgt))

        all_relationships.extend(batch_relationships)

    context = "\n".join([f"{s} - {r} -> {t}" for s, r, t in all_relationships])
    return context


structured_retriever_runnable = RunnableLambda(lambda x: structured_retriever_all(x["question"]))

graph_prompt = ChatPromptTemplate.from_template("""
You are given factual relationships extracted from a Neo4j knowledge graph.

Each relationship follows this format:
    Entity_A - RELATIONSHIP -> Entity_B

Your task:
1. Interpret these relationships to understand the underlying facts and context.
2. Use ONLY the provided context to answer the user's question.
3. Provide a clear, concise, and factual explanation.
4. If the context does not contain enough information to answer confidently, state that explicitly.

---

### Context:
{context}

### Question:
{question}

---

### Instructions for Response:
- Begin with a **direct answer** (1–3 sentences).
- Optionally include a **brief explanation** summarizing key entities or relationships that support your answer.
- Do NOT include speculative or external knowledge.
- Maintain a professional and academic tone suitable for expert analysis.

### Example Format:

Answer:
[Concise, factual response]

Supporting Evidence:
- [Entity_A] - [RELATIONSHIP] -> [Entity_B]
- [Entity_A] - [RELATIONSHIP] -> [Entity_B]

Confidence: [High | Medium | Low]

---

Now provide your answer below.
""")

chain = (
    {"context": structured_retriever_runnable, "question": RunnablePassthrough()}
    | graph_prompt
    | llm
    | StrOutputParser()
)



def get_answer(question: str) -> str:
    """Main entry function for Streamlit app."""
    try:
        return chain.invoke({"question": question})
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
