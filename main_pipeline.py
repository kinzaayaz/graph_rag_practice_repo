import os
import certifi
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field
from typing import List

load_dotenv()



# Load environment variables
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
    """Structured representation of entities or key concepts found in text."""
    names: List[str] = Field(
        ...,
        description=(
            "List of all distinct named entities, chemicals, technologies, processes, materials, "
            "parameters, or scientific terms mentioned in the text."
        ),
    )


entity_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert technical information extractor. "
            "Identify ALL key named entities and domain-specific terms from the text, "
            "including chemicals, units, parameters, costs, and technologies. "
            "Include both concrete (e.g., ammonia, PEM, electrolysis) and abstract terms (e.g., efficiency, storage, LCOA)."
        ),
    ),
    (
        "human",
        "Extract all distinct entities or parameters from this text:\n\n{question}"
    ),
])

# --- Chain the prompt with the model (structured output mode) ---
entity_chain = entity_prompt | llm.with_structured_output(Entities, method="function_calling")


from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars
def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

def structured_retriever_all(question: str, exclude_rels=None, batch_size=5):
    """
    Enhanced retriever: fetches direct + indirect relationships (depth 2)
    and relevant TableRow nodes.
    """
    exclude_rels = exclude_rels or ['MENTIONS']
    all_data = []
    query_limit = 10

    # --- Step 1: Extract entities from question ---
    entities = entity_chain.invoke({"question": question})
    entity_names = getattr(entities, "names", entities) if entities else []
    if isinstance(entity_names, str):
        entity_names = [entity_names]

    for i in range(0, len(entity_names), batch_size):
        batch = entity_names[i:i + batch_size]

        for entity in batch:
            try:
                # --- Step 2: Full-text search (depth 1 and 2) ---
                cypher = """
                CALL db.index.fulltext.queryNodes('entity', $query) YIELD node
                // Direct relationships
                OPTIONAL MATCH (node)-[r1]->(n1)
                WHERE NOT type(r1) IN $exclude_rels
                // Indirect (2-hop) relationships
                OPTIONAL MATCH (node)-[r2*2]-(n2)
                WHERE ALL(rel IN r2 WHERE NOT type(rel) IN $exclude_rels)
                WITH COLLECT(DISTINCT {source: node, rel: r1, target: n1}) +
                     COLLECT(DISTINCT {source: node, rels: r2, target: n2}) AS rels
                UNWIND rels AS data
                WITH data.source AS source,
                     (CASE WHEN data.rel IS NOT NULL THEN type(data.rel)
                           ELSE REDUCE(types = [], r IN data.rels | types + type(r))
                      END) AS rel_type,
                     data.target AS target
                RETURN DISTINCT
                    source.id AS source_id,
                    labels(source) AS source_labels,
                    rel_type,
                    target.id AS target_id,
                    labels(target) AS target_labels,
                    properties(source) AS source_props,
                    properties(target) AS target_props
                LIMIT $limit
                """
                rows = graph.query(
                    cypher,
                    {
                        "query": generate_full_text_query(entity),
                        "exclude_rels": exclude_rels,
                        "limit": query_limit
                    }
                )
            except Exception as e:
                print(f"⚠️ Full-text search failed for '{entity}': {e}")
                # --- Fallback: substring search ---
                cypher_fallback = """
                MATCH (n)
                WHERE toLower(n.id) CONTAINS toLower($q)
                OPTIONAL MATCH (n)-[r*1..2]-(m)
                WHERE ALL(rel IN r WHERE NOT type(rel) IN $exclude_rels)
                RETURN DISTINCT
                    n.id AS source_id,
                    labels(n) AS source_labels,
                    [rel IN r | type(rel)] AS rel_type,
                    m.id AS target_id,
                    labels(m) AS target_labels,
                    properties(n) AS source_props,
                    properties(m) AS target_props
                LIMIT $limit
                """
                rows = graph.query(
                    cypher_fallback,
                    {
                        "q": entity,
                        "exclude_rels": exclude_rels,
                        "limit": query_limit
                    }
                )

            for row in rows:
                rel_types = row.get("rel_type")
                if isinstance(rel_types, list):
                    rel = " -> ".join(rel_types)
                else:
                    rel = rel_types

                all_data.append({
                    "type": "relationship",
                    "source": row.get("source_id"),
                    "source_labels": row.get("source_labels", []),
                    "rel": rel,
                    "target": row.get("target_id"),
                    "target_labels": row.get("target_labels", []),
                    "source_props": row.get("source_props", {}),
                    "target_props": row.get("target_props", {})
                })

    # --- Step 3: TableRow retrieval (same as before) ---
    try:
        index_check = graph.query("SHOW FULLTEXT INDEXES")
        index_exists = any(idx.get('name') == 'tablerow_index' for idx in index_check)

        if index_exists:
            cypher_rows = """
            CALL db.index.fulltext.queryNodes('tablerow_index', $q) YIELD node, score
            RETURN node.id AS row_id, properties(node) AS row_props, score
            ORDER BY score DESC
            LIMIT $limit
            """
            table_rows_res = graph.query(
                cypher_rows,
                {"q": question, "limit": query_limit}
            )
        else:
            cypher_rows_fallback = """
            MATCH (r:TableRow)
            WHERE any(prop IN keys(r)
                WHERE toLower(prop) CONTAINS toLower($q)
                   OR toLower(toString(r[prop])) CONTAINS toLower($q))
            RETURN r.id AS row_id, properties(r) AS row_props
            LIMIT $limit
            """
            table_rows_res = graph.query(
                cypher_rows_fallback,
                {"q": question, "limit": query_limit}
            )

        for tr in table_rows_res:
            all_data.append({
                "type": "table_row",
                "row_id": tr.get("row_id"),
                "row_props": tr.get("row_props", {})
            })

    except Exception as e:
        print(f"⚠️ Table row retrieval failed: {e}")

    return all_data


structured_retriever_runnable = RunnableLambda(lambda x: structured_retriever_all(x["question"]))

graph_prompt = ChatPromptTemplate.from_template("""
You are an expert technical analyst. You are given context from a Neo4j knowledge graph:
- Entities, their properties (including numeric values and table rows)
- Relationships between entities

The context may include tables, figures, numeric values, and textual descriptions.

Your task:
1. Use the graph context to answer the user's question.
2. Extract answers directly from table rows, numeric values, or relationships wherever applicable.
3. Use ONLY the provided graph context.
4. Clearly state if any information is missing.

### Context:
{context}

### Question:
{question}

### Instructions for Response:
- Start with a **direct answer** (1–3 sentences)
- Include **ALL relationships from the graph context** under "Supporting Evidence"
- Include **table rows, numeric values, and properties** wherever relevant
- Mention the source of each piece of evidence (graph)
- Maintain a professional and factual tone
- Confidence: High/Medium/Low

### Output Format:

Answer:
[Concise, factual response]

Supporting Evidence:
- [Entity_A] - [RELATIONSHIP] -> [Entity_B]  (source: graph)
- Table Row / Numeric Value: [value1, value2, …]  (source: graph)
- Property: [key=value]  (source: graph)

Confidence: [High | Medium | Low]
""")


chain = (
    {"context": structured_retriever_runnable, "question": RunnablePassthrough()}
    | graph_prompt
    | llm
    | StrOutputParser()
)


def get_answer(question: str) -> str:
    try:
        return chain.invoke({"question": question})
    except Exception as e:
        return f"⚠️ Error: {e}"