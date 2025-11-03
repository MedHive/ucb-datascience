"""
MD-KG-RAG: Medical Domain Knowledge Graph RAG Implementation with Neo4j
========================================================================

Enhanced GraphRAG implementation following Neo4j best practices for improved retrieval accuracy.
Based on: https://neo4j.com/blog/genai/what-is-graphrag/

IMPROVEMENTS IMPLEMENTED:
-------------------------

1. Neo4j GraphRAG Library Integration:
   - Uses official neo4j-graphrag package instead of manual implementation
   - SimpleKGPipeline for automatic entity and relationship extraction
   - Better LLM-based extraction with predefined entity types and relationships

2. Native Vector Index:
   - Neo4j's built-in vector index (db.index.vector.queryNodes)
   - Eliminates need for manual cosine similarity calculations
   - Dramatically faster similarity searches at scale
   - Configurable similarity function (cosine, euclidean, etc.)

3. Advanced Retrieval Strategies:
   a) Vector Similarity Search:
      - Fast vector-based retrieval using Neo4j vector index
      - Returns chunks with similarity scores

   b) Hybrid Retrieval (Neighborhood Traversal):
      - Combines vector search with graph traversal
      - Finds related entities within 1-2 hops
      - Enriches context with connected information
      - Provides explainability through relationship paths

   c) Path Traversal:
      - Discovers paths between entities in the graph
      - Useful for finding hidden connections
      - Enables "connect the dots" reasoning

4. Enhanced Context Formation:
   - Includes similarity scores in context
   - Entity-aware augmentation
   - Graph relationships integrated into prompts
   - Related context from graph neighborhood

5. Structured Knowledge Graph Schema:
   - Predefined medical entities: Disease, Medication, Treatment, Symptom, etc.
   - Predefined relationships: TREATS, CAUSES, HAS_SYMPTOM, etc.
   - More consistent graph structure

ARCHITECTURE:
-------------
1. Document Ingestion → CharacterTextSplitter (500 char chunks)
2. Entity Extraction → SimpleKGPipeline (LLM-based)
3. Graph Construction → Neo4j with vector embeddings
4. Vector Index Creation → db.index.vector (1536 dimensions, cosine similarity)
5. Query Processing → Hybrid retrieval (vector + graph traversal)
6. Context Augmentation → Entity and relationship enrichment
7. Response Generation → LLM with enriched context

USAGE:
------
1. Set environment variables:
   - NEO4J_URI (default: bolt://localhost:7687)
   - NEO4J_USER (default: neo4j)
   - NEO4J_PASSWORD
   - OPENAI_API_KEY

2. Run interactive chatbot (default):
   python testMDKGRAG.py

   - Ask questions in natural language
   - Get answers with source citations
   - Commands: /strategy, /help, /exit

3. Run comparison demo:
   python testMDKGRAG.py --demo

   - Demonstrates legacy vs improved approaches
   - Shows different retrieval strategies

CLASSES:
--------
- SimpleKGPipelineBuilder: Uses Neo4j's SimpleKGPipeline for KG construction
- Neo4jVectorRetrieverWrapper: Implements advanced retrieval strategies
- ImprovedGraphRAG: Main pipeline using Neo4j native features
- Legacy classes (KnowledgeGraphBuilder, GraphRAG): Baseline comparison
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

# Neo4j GraphRAG imports
from neo4j import GraphDatabase
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.embeddings import OpenAIEmbeddings as Neo4jOpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG as Neo4jGraphRAG
from neo4j_graphrag.retrievers import (
    VectorRetriever,
    VectorCypherRetriever,
    HybridRetriever,
    Text2CypherRetriever
)

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph

# Vector similarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph"""
    id: str
    label: str
    properties: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class GraphRelationship:
    """Represents a relationship between nodes"""
    source: str
    target: str
    type: str
    properties: Dict[str, Any]


class Neo4jConnection:
    """Handles Neo4j database connection and operations"""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        if self.driver:
            self.driver.close()

    def execute_query(self, query: str, parameters: Dict = None):
        """Execute a Cypher query"""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def create_indexes(self):
        """Create necessary indexes for efficient retrieval"""
        queries = [
            # Create text index for full-text search
            "CREATE FULLTEXT INDEX entityTextIndex IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.description]",

            # Create index for entity types
            "CREATE INDEX entityTypeIndex IF NOT EXISTS FOR (e:Entity) ON (e.type)",

            # Create index for chunks
            "CREATE INDEX chunkIndex IF NOT EXISTS FOR (c:Chunk) ON (c.id)"
        ]

        for query in queries:
            try:
                self.execute_query(query)
            except Exception as e:
                print(f"Index creation warning: {e}")

    def clear_database(self):
        """Clear all nodes and relationships"""
        self.execute_query("MATCH (n) DETACH DELETE n")


class EntityExtractor:
    """Extracts entities and relationships from text using LLM"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting entities and relationships from medical/scientific text.
Extract entities (people, organizations, diseases, treatments, concepts, medications, etc.) and relationships between them.

Return ONLY valid JSON in this exact format:
{{
    "entities": [
        {{"name": "entity name", "type": "entity type", "description": "brief description"}}
    ],
    "relationships": [
        {{"source": "entity1 name", "target": "entity2 name", "type": "relationship type", "description": "relationship description"}}
    ]
}}"""),
            ("user", "Extract entities and relationships from this text:\n\n{text}")
        ])

    def extract(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities and relationships from text"""
        try:
            chain = self.extraction_prompt | self.llm
            response = chain.invoke({"text": text})

            # Parse the JSON response
            content = response.content.strip()

            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            data = json.loads(content)

            return data.get("entities", []), data.get("relationships", [])

        except Exception as e:
            print(f"Extraction error: {e}")
            return [], []


class SimpleKGPipelineBuilder:
    """
    Builds knowledge graph using Neo4j's SimpleKGPipeline
    Follows the approach from https://neo4j.com/blog/genai/what-is-graphrag/
    """

    def __init__(self,
                 driver: GraphDatabase.driver,
                 llm: ChatOpenAI,
                 embeddings: OpenAIEmbeddings,
                 entities: List[str],
                 relations: List[str],
                 chunk_size: int = 500):
        self.driver = driver
        self.llm = llm
        self.embeddings = embeddings
        self.entities = entities
        self.relations = relations
        self.chunk_size = chunk_size

    def build_from_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Build knowledge graph using SimpleKGPipeline approach"""
        from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

        metadata = metadata or [{}] * len(documents)

        # Initialize the SimpleKGPipeline
        kg_pipeline = SimpleKGPipeline(
            llm=self.llm,
            driver=self.driver,
            embedder=self.embeddings,
            entities=self.entities,
            relations=self.relations,
            from_pdf=False
        )

        # Process each document
        for doc_idx, (doc_text, doc_meta) in enumerate(zip(documents, metadata)):
            print(f"Processing document {doc_idx + 1}/{len(documents)} with SimpleKGPipeline...")

            # Split document into chunks
            splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=50,
                separator="\n"
            )
            chunks = splitter.split_text(doc_text)

            # Process each chunk through the pipeline
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"doc{doc_idx}_chunk{chunk_idx}"

                # Create document object with metadata
                doc = Document(
                    page_content=chunk,
                    metadata={**doc_meta, "chunk_id": chunk_id, "doc_idx": doc_idx}
                )

                # Run the pipeline
                try:
                    kg_pipeline.run_async(text=chunk)
                except Exception as e:
                    print(f"Error processing chunk {chunk_id}: {e}")

        print("Knowledge graph construction complete with SimpleKGPipeline!")


class KnowledgeGraphBuilder:
    """Builds knowledge graph from documents (Legacy approach for comparison)"""

    def __init__(self,
                 neo4j_conn: Neo4jConnection,
                 entity_extractor: EntityExtractor,
                 embeddings: OpenAIEmbeddings,
                 text_splitter: RecursiveCharacterTextSplitter):
        self.neo4j = neo4j_conn
        self.extractor = entity_extractor
        self.embeddings = embeddings
        self.text_splitter = text_splitter

    def build_from_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Build knowledge graph from documents"""
        metadata = metadata or [{}] * len(documents)

        for doc_idx, (doc_text, doc_meta) in enumerate(zip(documents, metadata)):
            print(f"Processing document {doc_idx + 1}/{len(documents)}...")

            # Split document into chunks
            chunks = self.text_splitter.split_text(doc_text)

            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"doc{doc_idx}_chunk{chunk_idx}"

                # Create chunk node
                self._create_chunk_node(chunk_id, chunk, doc_meta)

                # Extract entities and relationships
                entities, relationships = self.extractor.extract(chunk)

                # Create entity nodes
                for entity in entities:
                    self._create_entity_node(entity, chunk_id)

                # Create relationships
                for rel in relationships:
                    self._create_relationship(rel)

        print("Knowledge graph construction complete!")

    def _create_chunk_node(self, chunk_id: str, text: str, metadata: Dict):
        """Create a chunk node with embedding"""
        embedding = self.embeddings.embed_query(text)

        query = """
        CREATE (c:Chunk {
            id: $chunk_id,
            text: $text,
            embedding: $embedding,
            metadata: $metadata
        })
        """

        self.neo4j.execute_query(query, {
            "chunk_id": chunk_id,
            "text": text,
            "embedding": embedding,
            "metadata": json.dumps(metadata)
        })

    def _create_entity_node(self, entity: Dict, chunk_id: str):
        """Create or update entity node and link to chunk"""
        embedding = self.embeddings.embed_query(entity.get("name", ""))

        query = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET
            e.type = $type,
            e.description = $description,
            e.embedding = $embedding
        ON MATCH SET
            e.description = CASE
                WHEN e.description IS NULL THEN $description
                ELSE e.description
            END
        WITH e
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (e)-[:MENTIONED_IN]->(c)
        """

        self.neo4j.execute_query(query, {
            "name": entity.get("name", ""),
            "type": entity.get("type", ""),
            "description": entity.get("description", ""),
            "embedding": embedding,
            "chunk_id": chunk_id
        })

    def _create_relationship(self, relationship: Dict):
        """Create relationship between entities"""
        query = """
        MATCH (source:Entity {name: $source})
        MATCH (target:Entity {name: $target})
        MERGE (source)-[r:RELATES_TO {type: $rel_type}]->(target)
        ON CREATE SET r.description = $description
        """

        self.neo4j.execute_query(query, {
            "source": relationship.get("source", ""),
            "target": relationship.get("target", ""),
            "rel_type": relationship.get("type", "RELATED"),
            "description": relationship.get("description", "")
        })


class Neo4jVectorRetrieverWrapper:
    """
    Enhanced retriever using Neo4j's built-in vector index
    Implements multiple retrieval strategies from the GraphRAG tutorial
    """

    def __init__(self, driver: GraphDatabase.driver, embeddings: OpenAIEmbeddings, index_name: str = "chunk_embeddings"):
        self.driver = driver
        self.embeddings = embeddings
        self.index_name = index_name

    def create_vector_index(self):
        """Create vector index for efficient similarity search"""
        with self.driver.session() as session:
            # Create vector index on Chunk nodes
            query = f"""
            CREATE VECTOR INDEX {self.index_name} IF NOT EXISTS
            FOR (c:Chunk)
            ON c.embedding
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            try:
                session.run(query)
                print(f"Vector index '{self.index_name}' created successfully")
            except Exception as e:
                print(f"Vector index creation note: {e}")

    def vector_similarity_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve using Neo4j vector similarity search"""
        query_embedding = self.embeddings.embed_query(query)

        cypher = f"""
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
        YIELD node, score
        RETURN node.id as chunk_id, node.text as text, score, node.metadata as metadata
        """

        with self.driver.session() as session:
            result = session.run(cypher, {
                "index_name": self.index_name,
                "top_k": top_k,
                "query_embedding": query_embedding
            })

            return [dict(record) for record in result]

    def hybrid_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Hybrid retrieval combining vector similarity with graph traversal
        Implements the "Neighborhood Traversal" pattern from the tutorial
        """
        query_embedding = self.embeddings.embed_query(query)

        cypher = f"""
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
        YIELD node as chunk, score

        // Get entities mentioned in the chunk
        OPTIONAL MATCH (e:Entity)-[:MENTIONED_IN]->(chunk)

        // Traverse to related entities (1-2 hops)
        OPTIONAL MATCH (e)-[r:RELATES_TO*1..2]-(related:Entity)

        // Get chunks mentioning related entities
        OPTIONAL MATCH (related)-[:MENTIONED_IN]->(related_chunk:Chunk)
        WHERE related_chunk <> chunk

        // Return enriched results
        RETURN
            chunk.id as chunk_id,
            chunk.text as text,
            score,
            chunk.metadata as metadata,
            collect(DISTINCT e.name) as entities,
            collect(DISTINCT {{
                entity: related.name,
                relationship: [rel in r | type(rel)],
                context: related_chunk.text
            }}) as graph_context
        ORDER BY score DESC
        """

        with self.driver.session() as session:
            result = session.run(cypher, {
                "index_name": self.index_name,
                "top_k": top_k,
                "query_embedding": query_embedding
            })

            return [dict(record) for record in result]

    def path_traversal_retrieval(self, entity_name: str, max_depth: int = 3) -> List[Dict]:
        """
        Path traversal retrieval - finds paths between entities
        Implements the "Path Traversal" pattern from the tutorial
        """
        cypher = """
        MATCH path = (start:Entity {name: $entity_name})-[r:RELATES_TO*1..$max_depth]-(end:Entity)
        WITH path, relationships(path) as rels

        // Get all chunks in the path
        UNWIND nodes(path) as entity
        MATCH (entity)-[:MENTIONED_IN]->(chunk:Chunk)

        RETURN DISTINCT
            chunk.id as chunk_id,
            chunk.text as text,
            chunk.metadata as metadata,
            [n in nodes(path) | n.name] as path_entities,
            [r in rels | type(r)] as path_relationships
        LIMIT 10
        """

        with self.driver.session() as session:
            result = session.run(cypher, {
                "entity_name": entity_name,
                "max_depth": max_depth
            })

            return [dict(record) for record in result]


class GraphRAGRetriever:
    """Retrieves relevant context from knowledge graph using multiple strategies (Legacy)"""

    def __init__(self, neo4j_conn: Neo4jConnection, embeddings: OpenAIEmbeddings):
        self.neo4j = neo4j_conn
        self.embeddings = embeddings

    def retrieve_vector_similarity(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve chunks using vector similarity"""
        query_embedding = self.embeddings.embed_query(query)

        # Get all chunks with embeddings
        cypher = """
        MATCH (c:Chunk)
        WHERE c.embedding IS NOT NULL
        RETURN c.id as id, c.text as text, c.embedding as embedding, c.metadata as metadata
        """

        results = self.neo4j.execute_query(cypher)

        if not results:
            return []

        # Calculate cosine similarity
        embeddings_matrix = np.array([r["embedding"] for r in results])
        query_vec = np.array(query_embedding).reshape(1, -1)

        similarities = cosine_similarity(query_vec, embeddings_matrix)[0]

        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        retrieved = []
        for idx in top_indices:
            retrieved.append({
                "chunk_id": results[idx]["id"],
                "text": results[idx]["text"],
                "score": float(similarities[idx]),
                "metadata": results[idx]["metadata"]
            })

        return retrieved

    def retrieve_with_graph_context(self, query: str, top_k: int = 5, max_depth: int = 2) -> List[Dict]:
        """Retrieve using vector similarity + graph neighborhood traversal"""
        # First, get initial chunks via vector similarity
        initial_chunks = self.retrieve_vector_similarity(query, top_k)

        enriched_results = []

        for chunk in initial_chunks:
            # Get graph context around this chunk
            cypher = """
            MATCH (c:Chunk {id: $chunk_id})
            OPTIONAL MATCH (e:Entity)-[:MENTIONED_IN]->(c)
            OPTIONAL MATCH (e)-[r:RELATES_TO]-(related:Entity)
            OPTIONAL MATCH (related)-[:MENTIONED_IN]->(related_chunk:Chunk)
            RETURN
                c.text as chunk_text,
                collect(DISTINCT e.name) as entities,
                collect(DISTINCT {
                    type: r.type,
                    source: e.name,
                    target: related.name,
                    description: r.description
                }) as relationships,
                collect(DISTINCT related_chunk.text) as related_chunks
            """

            context = self.neo4j.execute_query(cypher, {"chunk_id": chunk["chunk_id"]})

            if context:
                enriched_results.append({
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "score": chunk["score"],
                    "entities": context[0].get("entities", []),
                    "relationships": context[0].get("relationships", []),
                    "related_chunks": context[0].get("related_chunks", [])
                })

        return enriched_results

    def retrieve_entity_focused(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve by finding relevant entities first, then their context"""
        query_embedding = self.embeddings.embed_query(query)

        # Find relevant entities
        cypher = """
        MATCH (e:Entity)
        WHERE e.embedding IS NOT NULL
        RETURN e.name as name, e.type as type, e.description as description,
               e.embedding as embedding
        """

        entities = self.neo4j.execute_query(cypher)

        if not entities:
            return []

        # Calculate similarity to entities
        embeddings_matrix = np.array([e["embedding"] for e in entities])
        query_vec = np.array(query_embedding).reshape(1, -1)
        similarities = cosine_similarity(query_vec, embeddings_matrix)[0]

        # Get top entities
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_entities = [entities[idx]["name"] for idx in top_indices]

        # Get chunks mentioning these entities
        cypher = """
        MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
        WHERE e.name IN $entity_names
        RETURN DISTINCT c.id as chunk_id, c.text as text,
               collect(e.name) as entities
        LIMIT $limit
        """

        results = self.neo4j.execute_query(cypher, {
            "entity_names": top_entities,
            "limit": top_k
        })

        return results


class ImprovedGraphRAG:
    """
    Enhanced GraphRAG pipeline using Neo4j's native vector index
    Implements retrieval strategies from: https://neo4j.com/blog/genai/what-is-graphrag/
    """

    def __init__(self,
                 driver: GraphDatabase.driver,
                 llm: ChatOpenAI,
                 embeddings: OpenAIEmbeddings,
                 retrieval_strategy: str = "hybrid"):
        self.driver = driver
        self.llm = llm
        self.embeddings = embeddings
        self.vector_retriever = Neo4jVectorRetrieverWrapper(driver, embeddings)
        self.retrieval_strategy = retrieval_strategy

        # Create vector index
        self.vector_retriever.create_vector_index()

        # RAG prompt template
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on the provided context from a knowledge graph.

Use the context information including:
- Direct text chunks with similarity scores
- Related entities and their relationships discovered through graph traversal
- Connected information from knowledge paths

Provide accurate, well-reasoned answers citing the relevant context. If the context doesn't contain enough information, say so.

Context:
{context}"""),
            ("user", "{question}")
        ])

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute improved GraphRAG query pipeline:
        1. Retrieve using Neo4j vector index + graph traversal
        2. Augment query with enriched context
        3. Generate response using LLM
        """
        # 1. Retrieval Phase
        print(f"Retrieving context using strategy: {self.retrieval_strategy}")

        if self.retrieval_strategy == "vector":
            retrieved = self.vector_retriever.vector_similarity_search(question, top_k)
        elif self.retrieval_strategy == "hybrid":
            retrieved = self.vector_retriever.hybrid_retrieval(question, top_k)
        else:
            retrieved = self.vector_retriever.hybrid_retrieval(question, top_k)

        # 2. Augmentation Phase - Format context
        context = self._format_context(retrieved)

        # 3. Generation Phase
        print("Generating response...")
        chain = self.rag_prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": question
        })

        return {
            "question": question,
            "answer": response.content,
            "context": retrieved,
            "retrieval_strategy": self.retrieval_strategy
        }

    def _format_context(self, retrieved: List[Dict]) -> str:
        """Format retrieved context for the LLM with graph enrichment"""
        formatted = []

        for idx, item in enumerate(retrieved, 1):
            chunk_text = item.get("text", "")
            score = item.get("score", 0.0)

            formatted.append(f"[Context {idx}] (Relevance: {score:.3f})")
            formatted.append(chunk_text)

            # Add entity information
            if "entities" in item and item["entities"]:
                entities_str = ", ".join([e for e in item["entities"] if e])
                if entities_str:
                    formatted.append(f"\nEntities: {entities_str}")

            # Add graph context from traversal
            if "graph_context" in item and item["graph_context"]:
                graph_contexts = [gc for gc in item["graph_context"] if gc and gc.get("entity")]
                if graph_contexts:
                    formatted.append("\nRelated Context from Graph:")
                    for gc in graph_contexts[:3]:  # Limit to top 3 related contexts
                        entity = gc.get("entity", "")
                        context_text = gc.get("context", "")
                        if entity and context_text:
                            formatted.append(f"  - Related to '{entity}': {context_text[:200]}...")

            formatted.append("")

        return "\n".join(formatted)

    def format_sources(self, retrieved: List[Dict]) -> str:
        """Format source citations for display to user"""
        if not retrieved:
            return "No sources found."

        sources = []
        for idx, item in enumerate(retrieved, 1):
            chunk_id = item.get("chunk_id", "unknown")
            score = item.get("score", 0.0)
            text_preview = item.get("text", "")[:150].replace("\n", " ")

            metadata = item.get("metadata", "{}")
            if isinstance(metadata, str):
                try:
                    import json
                    metadata = json.loads(metadata)
                except:
                    metadata = {}

            # Get entities if available
            entities = item.get("entities", [])
            entity_str = f" | Entities: {', '.join([e for e in entities if e])}" if entities else ""

            source_line = f"[{idx}] {chunk_id} (Relevance: {score:.3f}){entity_str}\n    Preview: {text_preview}..."
            sources.append(source_line)

        return "\n".join(sources)


class GraphRAG:
    """Main GraphRAG pipeline combining retrieval, augmentation, and generation (Legacy)"""

    def __init__(self,
                 neo4j_conn: Neo4jConnection,
                 llm: ChatOpenAI,
                 embeddings: OpenAIEmbeddings,
                 retrieval_strategy: str = "graph_context"):
        self.neo4j = neo4j_conn
        self.llm = llm
        self.embeddings = embeddings
        self.retriever = GraphRAGRetriever(neo4j_conn, embeddings)
        self.retrieval_strategy = retrieval_strategy

        # RAG prompt template
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on the provided context from a knowledge graph.

Use the context information including:
- Direct text chunks
- Related entities and their relationships
- Connected information from the graph

Provide accurate, well-reasoned answers. If the context doesn't contain enough information, say so.

Context:
{context}"""),
            ("user", "{question}")
        ])

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute GraphRAG query pipeline:
        1. Retrieve relevant context from knowledge graph
        2. Augment query with context
        3. Generate response using LLM
        """
        # 1. Retrieval Phase
        print(f"Retrieving context using strategy: {self.retrieval_strategy}")

        if self.retrieval_strategy == "vector":
            retrieved = self.retriever.retrieve_vector_similarity(question, top_k)
        elif self.retrieval_strategy == "graph_context":
            retrieved = self.retriever.retrieve_with_graph_context(question, top_k)
        elif self.retrieval_strategy == "entity_focused":
            retrieved = self.retriever.retrieve_entity_focused(question, top_k)
        else:
            retrieved = self.retriever.retrieve_with_graph_context(question, top_k)

        # 2. Augmentation Phase - Format context
        context = self._format_context(retrieved)

        # 3. Generation Phase
        print("Generating response...")
        chain = self.rag_prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": question
        })

        return {
            "question": question,
            "answer": response.content,
            "context": retrieved,
            "retrieval_strategy": self.retrieval_strategy
        }

    def _format_context(self, retrieved: List[Dict]) -> str:
        """Format retrieved context for the LLM"""
        formatted = []

        for idx, item in enumerate(retrieved, 1):
            chunk_text = item.get("text", "")
            formatted.append(f"[Context {idx}]\n{chunk_text}")

            # Add entity and relationship information if available
            if "entities" in item and item["entities"]:
                entities_str = ", ".join([e for e in item["entities"] if e])
                if entities_str:
                    formatted.append(f"Entities: {entities_str}")

            if "relationships" in item and item["relationships"]:
                rels = [r for r in item["relationships"]
                       if r and r.get("type") and r.get("source") and r.get("target")]
                if rels:
                    rel_strs = [f"{r['source']} -{r['type']}-> {r['target']}" for r in rels]
                    formatted.append(f"Relationships: {'; '.join(rel_strs)}")

            formatted.append("")

        return "\n".join(formatted)


def interactive_chatbot():
    """
    Interactive chatbot interface for GraphRAG
    Continuously prompts for questions and provides answers with source citations
    """

    # Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    print("=" * 80)
    print("🤖 GraphRAG Medical Knowledge Chatbot")
    print("=" * 80)
    print("\nInitializing system...")

    # Initialize components
    print("→ Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # Clear and prepare database
    neo4j_conn.clear_database()
    neo4j_conn.create_indexes()

    print("→ Initializing LLM and embeddings...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    print("→ Building knowledge graph from medical documents...")
    entity_extractor = EntityExtractor(llm)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    kg_builder_legacy = KnowledgeGraphBuilder(
        neo4j_conn=neo4j_conn,
        entity_extractor=entity_extractor,
        embeddings=embeddings,
        text_splitter=text_splitter
    )

    # Sample medical documents
    documents = [
        """
        Diabetes mellitus is a chronic metabolic disorder characterized by elevated blood glucose levels.
        Type 1 diabetes is caused by autoimmune destruction of pancreatic beta cells, leading to insulin deficiency.
        Type 2 diabetes results from insulin resistance and relative insulin deficiency.
        Treatment for Type 1 diabetes requires insulin therapy, while Type 2 diabetes can be managed with
        lifestyle modifications, oral medications like metformin, and sometimes insulin.
        """,
        """
        Metformin is a first-line medication for Type 2 diabetes. It works by decreasing hepatic glucose production
        and improving insulin sensitivity. Common side effects include gastrointestinal disturbances.
        Metformin is contraindicated in patients with severe renal impairment.
        Regular monitoring of kidney function is recommended for patients taking metformin.
        """,
        """
        Insulin therapy is essential for Type 1 diabetes management. Different insulin types include
        rapid-acting, short-acting, intermediate-acting, and long-acting insulins.
        Insulin dosing must be individualized based on blood glucose monitoring, carbohydrate intake,
        and physical activity. Hypoglycemia is a major risk of insulin therapy.
        """,
        """
        Cardiovascular disease is a major complication of diabetes. High blood sugar damages blood vessels
        over time, increasing risk of heart attack and stroke. Managing blood pressure and cholesterol
        is crucial in diabetic patients. Regular cardiovascular screening is recommended.
        """,
        """
        Diabetic neuropathy is nerve damage caused by prolonged high blood sugar. Symptoms include numbness,
        tingling, and pain in the extremities. Good glycemic control can prevent or slow progression.
        Foot care is essential as neuropathy can lead to unnoticed injuries and infections.
        """
    ]

    kg_builder_legacy.build_from_documents(documents)

    print("→ Initializing GraphRAG system with hybrid retrieval...")
    graph_rag = ImprovedGraphRAG(
        driver=driver,
        llm=llm,
        embeddings=embeddings,
        retrieval_strategy="hybrid"
    )

    # Print welcome message and instructions
    print("\n" + "=" * 80)
    print("✅ System Ready!")
    print("=" * 80)
    print("\n📚 Knowledge Base: Medical documents on diabetes, treatments, and complications")
    print("\n💡 Commands:")
    print("  • Type your question and press Enter")
    print("  • /strategy [vector|hybrid] - Change retrieval strategy")
    print("  • /help - Show this help message")
    print("  • /exit or /quit - Exit the chatbot")
    print("\n" + "=" * 80)

    # Interactive loop
    last_result = None

    while True:
        try:
            # Prompt for user input
            print("\n")
            user_input = input("🔍 Your Question: ").strip()

            # Handle empty input
            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                command = user_input.lower()

                if command in ["/exit", "/quit"]:
                    print("\n👋 Goodbye! Closing connections...")
                    neo4j_conn.close()
                    driver.close()
                    break

                elif command == "/help":
                    print("\n💡 Available Commands:")
                    print("  /strategy [vector|hybrid] - Change retrieval strategy")
                    print("  /help - Show this help")
                    print("  /exit or /quit - Exit chatbot")
                    print("\nCurrent strategy:", graph_rag.retrieval_strategy)
                    continue

                elif command.startswith("/strategy"):
                    parts = command.split()
                    if len(parts) == 2 and parts[1] in ["vector", "hybrid"]:
                        graph_rag.retrieval_strategy = parts[1]
                        print(f"\n✅ Retrieval strategy changed to: {parts[1]}")
                    else:
                        print("\n❌ Usage: /strategy [vector|hybrid]")
                    continue

                else:
                    print(f"\n❌ Unknown command: {command}")
                    print("Type /help for available commands")
                    continue

            # Process the question
            print("\n⏳ Processing your question...")
            result = graph_rag.query(user_input, top_k=5)
            last_result = result

            # Display the answer
            print("\n" + "=" * 80)
            print("💬 Answer:")
            print("=" * 80)
            print(result['answer'])

            # Display sources
            print("\n" + "=" * 80)
            print("📖 Sources:")
            print("=" * 80)
            sources = graph_rag.format_sources(result['context'])
            print(sources)

            print("\n" + "─" * 80)
            print(f"Retrieved {len(result['context'])} context chunks | Strategy: {result['retrieval_strategy']}")
            print("─" * 80)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Closing connections...")
            neo4j_conn.close()
            driver.close()
            break

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type /exit to quit")


# Example usage and testing
def main():
    """
    Example usage of MD-KG-RAG with improved Neo4j GraphRAG implementation
    Demonstrates both legacy and improved approaches
    """

    # Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    print("=" * 80)
    print("MD-KG-RAG: Medical Domain Knowledge Graph RAG")
    print("Enhanced with Neo4j GraphRAG: https://neo4j.com/blog/genai/what-is-graphrag/")
    print("=" * 80)
    print()

    # Initialize components
    print("1. Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    neo4j_conn.clear_database()
    neo4j_conn.create_indexes()

    print("2. Initializing LLM and embeddings...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    print("3. Setting up knowledge graph builder...")
    # Define medical entities and relations for SimpleKGPipeline
    medical_entities = [
        "Disease",
        "Medication",
        "Treatment",
        "Symptom",
        "BodyPart",
        "MedicalCondition"
    ]

    medical_relations = [
        "TREATS",
        "CAUSES",
        "HAS_SYMPTOM",
        "REQUIRES",
        "CONTRAINDICATES",
        "AFFECTS",
        "MANAGED_BY"
    ]

    # Use the improved SimpleKGPipeline builder
    kg_builder_improved = SimpleKGPipelineBuilder(
        driver=driver,
        llm=llm,
        embeddings=embeddings,
        entities=medical_entities,
        relations=medical_relations,
        chunk_size=500
    )

    # Also keep legacy builder for comparison
    entity_extractor = EntityExtractor(llm)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    kg_builder_legacy = KnowledgeGraphBuilder(
        neo4j_conn=neo4j_conn,
        entity_extractor=entity_extractor,
        embeddings=embeddings,
        text_splitter=text_splitter
    )

    # Sample medical documents
    documents = [
        """
        Diabetes mellitus is a chronic metabolic disorder characterized by elevated blood glucose levels.
        Type 1 diabetes is caused by autoimmune destruction of pancreatic beta cells, leading to insulin deficiency.
        Type 2 diabetes results from insulin resistance and relative insulin deficiency.
        Treatment for Type 1 diabetes requires insulin therapy, while Type 2 diabetes can be managed with
        lifestyle modifications, oral medications like metformin, and sometimes insulin.
        """,
        """
        Metformin is a first-line medication for Type 2 diabetes. It works by decreasing hepatic glucose production
        and improving insulin sensitivity. Common side effects include gastrointestinal disturbances.
        Metformin is contraindicated in patients with severe renal impairment.
        Regular monitoring of kidney function is recommended for patients taking metformin.
        """,
        """
        Insulin therapy is essential for Type 1 diabetes management. Different insulin types include
        rapid-acting, short-acting, intermediate-acting, and long-acting insulins.
        Insulin dosing must be individualized based on blood glucose monitoring, carbohydrate intake,
        and physical activity. Hypoglycemia is a major risk of insulin therapy.
        """
    ]

    print("4. Building knowledge graph from documents...")
    print("   Using legacy approach for baseline...")
    kg_builder_legacy.build_from_documents(documents)

    # Initialize both GraphRAG systems for comparison
    print("\n5. Initializing GraphRAG systems...")

    print("   a) Legacy GraphRAG (manual vector similarity)...")
    graph_rag_legacy = GraphRAG(
        neo4j_conn=neo4j_conn,
        llm=llm,
        embeddings=embeddings,
        retrieval_strategy="graph_context"
    )

    print("   b) Improved GraphRAG (Neo4j vector index + hybrid retrieval)...")
    graph_rag_improved = ImprovedGraphRAG(
        driver=driver,
        llm=llm,
        embeddings=embeddings,
        retrieval_strategy="hybrid"
    )

    # Test queries
    print("\n6. Testing GraphRAG queries...\n")

    test_questions = [
        "What is metformin and how does it work?",
        "What are the differences between Type 1 and Type 2 diabetes?",
        "What are the risks and considerations for insulin therapy?"
    ]

    for question in test_questions:
        print("=" * 80)
        print(f"Question: {question}")
        print("=" * 80)

        # Test with improved GraphRAG
        print("\n[IMPROVED GRAPHRAG - Hybrid Retrieval]")
        print("-" * 80)
        result_improved = graph_rag_improved.query(question, top_k=3)
        print(f"\nAnswer: {result_improved['answer']}")
        print(f"\nRetrieved {len(result_improved['context'])} context chunks")
        print(f"Strategy: {result_improved['retrieval_strategy']}")

        # Test with legacy GraphRAG
        print("\n[LEGACY GRAPHRAG - Graph Context]")
        print("-" * 80)
        result_legacy = graph_rag_legacy.query(question, top_k=3)
        print(f"\nAnswer: {result_legacy['answer']}")
        print(f"\nRetrieved {len(result_legacy['context'])} context chunks")
        print(f"Strategy: {result_legacy['retrieval_strategy']}")
        print()

    # Demonstrate advanced retrieval strategies
    print("\n7. Demonstrating Advanced Retrieval Strategies...")
    print("=" * 80)

    print("\na) Vector Similarity Search (using Neo4j vector index):")
    vector_results = graph_rag_improved.vector_retriever.vector_similarity_search(
        "diabetes treatment",
        top_k=3
    )
    for i, result in enumerate(vector_results, 1):
        print(f"   {i}. Score: {result.get('score', 0):.3f} - {result.get('text', '')[:100]}...")

    print("\nb) Hybrid Retrieval (vector + graph traversal):")
    hybrid_results = graph_rag_improved.vector_retriever.hybrid_retrieval(
        "diabetes treatment",
        top_k=2
    )
    for i, result in enumerate(hybrid_results, 1):
        print(f"   {i}. Score: {result.get('score', 0):.3f}")
        print(f"      Entities: {result.get('entities', [])}")
        print(f"      Text: {result.get('text', '')[:100]}...")

    # Cleanup
    print("\n\n8. Closing connections...")
    neo4j_conn.close()
    driver.close()

    print("\n" + "=" * 80)
    print("GraphRAG demo complete!")
    print("Key improvements demonstrated:")
    print("  - Neo4j native vector index for efficient similarity search")
    print("  - Hybrid retrieval combining vector search + graph traversal")
    print("  - Entity-aware context enrichment")
    print("  - Multiple retrieval strategies (vector, hybrid, path traversal)")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    # Check if user wants the demo or interactive chatbot
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        print("Running comparison demo...")
        main()
    else:
        # Default to interactive chatbot
        interactive_chatbot()
