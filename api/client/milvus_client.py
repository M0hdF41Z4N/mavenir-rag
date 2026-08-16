# Vector database client. Manages the corpus collection schema, indexes, and hybrid
# search that combines dense embeddings (HNSW cosine) with two BM25 sparse indexes —
# one over the chunk body and one over the heading path — so heading-path hits are
# ranked separately and can be weighted higher than body-only matches.
import json
from pathlib import Path

from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    WeightedRanker,
    connections,
    utility,
)
from pymilvus.orm.mutation import MutationResult

from api.config import MilvusSettings
from api.models import (
    DocumentIDTopicPair,
    MilvusInsertResponse,
    MilvusSearchHit,
    TopicEnum,
)


class MilvusClient:
    def __init__(
        self,
        settings: MilvusSettings,
    ) -> None:
        """
        Initialize Milvus client.

        Connects to Milvus, creates the collection and index if they don't exist,
        and loads the collection into memory.

        Args:
            settings (MilvusSettings): Configuration object.
        """

        self.config = settings
        self.host = self.config.host
        self.port = self.config.port
        self.corpus_collection_name = self.config.corpus_collection_name
        self.vector_dim = self.config.vector_dim

        # Connect to the Milvus standalone instance running locally in Docker Container
        # Ref: https://milvus.io/api-reference/pymilvus/v2.4.x/ORM/Connections/connect.md#connect
        connections.connect("default", host=self.host, port=self.port, timeout=5)

        # Create or migrate the collection.
        # If the collection exists but is missing any BM25 sparse field (old schema),
        # drop it so it gets recreated with the updated schema that supports BM25 hybrid search
        # over both chunk body and heading path.
        if utility.has_collection(self.corpus_collection_name):
            existing = Collection(self.corpus_collection_name)
            field_names = {f.name for f in existing.schema.fields}
            if not {"sparse_vector", "headings_sparse_vector"}.issubset(field_names):
                existing.drop()

        if not utility.has_collection(self.corpus_collection_name):
            self._create_collection(self.corpus_collection_name)

        # Fetch the collection from the name
        # Ref: https://milvus.io/api-reference/pymilvus/v2.2.x/Collection/Collection().md#Collection
        self.corpus_collection = Collection(self.corpus_collection_name)

        # Create Index If Needed
        # Each Milvus collection must have an index else Milvus throws an error
        if len(self.corpus_collection.indexes) == 0:
            self._create_index()

        # Load Collection into memory for search or query
        # Ref: https://milvus.io/api-reference/pymilvus/v2.2.x/Collection/load().md#load
        self.corpus_collection.load(timeout=10)

    def _create_collection(self, collection_name: str) -> None:
        """
        Create a new collection in Milvus.

        Uses the vector dimension from self.config to define the FLOAT_VECTOR field.
        The primary key is an INT64 "id" field.
        """

        # Define the collection fields:
        # 'id' uniquely identifies each vector (e.g., a document chunk or item)
        # 'embedding' stores the vector representation of the data with dimension `vector_dim`
        # TODO: Add more fields to store metadata for the given embedding (e.g: Of which PDF file of docx it was originally a part of, etc.)
        # Ref: https://milvus.io/api-reference/pymilvus/v2.3.x/ORM/FieldSchema/FieldSchema.md#FieldSchema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
            # sparse_vector is auto-populated by the BM25 function defined on chunk_text
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            # headings_sparse_vector is auto-populated by a BM25 function defined on
            # headings_text — gives an exact heading-path hit its own rankable signal
            # so queries matching a heading outrank body-only matches.
            FieldSchema(name="headings_sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(
                name="chunk_text",
                dtype=DataType.VARCHAR,
                max_length=65535,  # Important: allow large chunks
                enable_analyzer=True,  # Required for BM25 tokenization
            ),
            FieldSchema(
                name="parent_headings",
                dtype=DataType.VARCHAR,
                max_length=2048,  # JSON-encoded list of heading strings
            ),
            FieldSchema(
                name="headings_text",
                dtype=DataType.VARCHAR,
                max_length=2048,
                enable_analyzer=True,  # Required for BM25 tokenization
            ),
            FieldSchema(
                name="page_num",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="topic",
                dtype=DataType.VARCHAR,
                max_length=64,
            ),
        ]

        # Create a collection schema using the defined fields
        # Ref: https://milvus.io/api-reference/pymilvus/v2.3.x/ORM/CollectionSchema/CollectionSchema.md#CollectionSchema
        schema = CollectionSchema(fields, description="Embeddings collection")

        # Add BM25 functions: Milvus auto-generates sparse vectors at insert time.
        # One for the chunk body, one for the heading path — scored independently so
        # the ranker can weight heading matches higher than body matches.
        # Ref: https://milvus.io/docs/full-text-search.md
        bm25_function = Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["chunk_text"],
            output_field_names=["sparse_vector"],
        )
        schema.add_function(bm25_function)

        headings_bm25_function = Function(
            name="headings_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["headings_text"],
            output_field_names=["headings_sparse_vector"],
        )
        schema.add_function(headings_bm25_function)

        # Initialize a Collection object to interact with the Milvus collection
        # Ref: https://milvus.io/api-reference/pymilvus/v2.2.x/Collection/Collection().md#Collection
        Collection(collection_name, schema)

    def _create_index(self) -> None:
        """
        - Create an index for the collection.
        - Create an index on the embedding field to speed up similarity search.
        - Index is like a shortcut that lets Milvus quickly find nearest vectors instead of scanning all the vectors during similarity search.
        - Uses the metric type, index type, and index-specific parameters from the self.config.

        Ref: https://milvus.io/api-reference/pymilvus/v2.2.x/Index/Index().md
        """
        # Ref: https://milvus.io/docs/index-vector-fields.md#Overview
        index_params = {
            "metric_type": self.config.metric_type,  # How similarity is measured (e.g., COSINE, L2)
            "index_type": self.config.index_type,  # Indexing algorithm (e.g., HNSW, IVF)
            "params": {
                "M": self.config.M,  # HNSW: number of neighbors per node (trade-off accuracy vs speed)
                "efConstruction": self.config.ef_construction,  # HNSW: how thoroughly the graph is built (higher = more accurate)
            },
        }

        # Ref: https://milvus.io/api-reference/pymilvus/v2.2.x/Collection/create_index().md#createindex
        self.corpus_collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )

        # Add sparse inverted index for BM25 full-text search
        # Ref: https://milvus.io/docs/full-text-search.md
        self.corpus_collection.create_index(
            field_name="sparse_vector",
            index_params={
                "metric_type": "BM25",
                "index_type": "SPARSE_INVERTED_INDEX",
            },
        )
        self.corpus_collection.create_index(
            field_name="headings_sparse_vector",
            index_params={
                "metric_type": "BM25",
                "index_type": "SPARSE_INVERTED_INDEX",
            },
        )

    def insert(
        self,
        vectors: list[list[float]],
        doc_ids: list[str],
        file_names: list[str],
        chunk_texts: list[str],
        parent_headings: list[str],
        headings_texts: list[str],
        page_nums: list[int],
        topics: list[TopicEnum],
    ) -> MilvusInsertResponse:
        if not (len(vectors) == len(doc_ids) == len(file_names) == len(chunk_texts) == len(parent_headings) == len(headings_texts) == len(page_nums)):
            raise ValueError("Failed to insert parsed document into Milvus: All inputs must have same length.")

        # Remove the last .md extension to recover the original filename
        cleaned_file_names = [Path(name).stem for name in file_names]

        result = self.corpus_collection.insert(
            [
                vectors,
                doc_ids,
                cleaned_file_names,
                chunk_texts,
                parent_headings,
                headings_texts,
                page_nums,
                [topic.value for topic in topics],
            ]
        )

        self.corpus_collection.flush()

        response = MilvusInsertResponse(
            insert_count=result.insert_count,
            delete_count=result.delete_count,
            upsert_count=result.upsert_count,
            success_count=result.succ_count,
            error_count=result.err_count,
            timestamp=result.timestamp,
            primary_keys=result.primary_keys,
        )

        return response

    def search(
        self,
        query_vectors: list[list[float]],
        query_texts: list[str],
        limit: int = 5,
        group_size: int = 1,
        expr: str | None = None,
    ) -> list[list[MilvusSearchHit]]:
        """
        Hybrid search combining dense vector similarity (HNSW) and two BM25
        full-text searches (one over the chunk body, one over the heading path),
        fused with a weighted ranker so heading-path hits outrank body-only hits.

        Args:
            query_vectors (List[List[float]]): Dense embedding vectors for each query.
            query_texts (List[str]): Raw query strings used for BM25 sparse search.
            limit (int, optional): Maximum number of top matching documents (grouped by doc_id). Defaults to 5.
            group_size (int, optional): Maximum chunks per document. Defaults to 1.
            expr (str, optional): Optional filter expression (e.g. `topic == "aviation"`).

        Returns:
            List[List[MilvusSearchHit]]: One list per query, each containing matched chunk hits.
        """
        output = []

        # Dense search params (HNSW cosine)
        dense_params: dict = {"metric_type": self.config.metric_type, "params": {}}
        if self.config.index_type == "HNSW":
            dense_params["params"] = {"ef": self.config.hnsw_ef}
        elif "IVF" in self.config.index_type:
            dense_params["params"] = {"nprobe": 10}

        output_fields = [
            "doc_id",
            "file_name",
            "chunk_text",
            "parent_headings",
            "page_num",
        ]

        # Run hybrid search per query (Milvus hybrid_search takes one query at a time)
        # Ref: https://milvus.io/docs/multi-vector-search.md
        for query_vector, query_text in zip(query_vectors, query_texts, strict=True):
            dense_req = AnnSearchRequest(
                data=[query_vector],
                anns_field="embedding",
                param=dense_params,
                limit=limit * group_size * self.config.search_over_fetch_factor,  # over-fetch before fusion
                expr=expr,
            )

            sparse_req = AnnSearchRequest(
                data=[query_text],
                anns_field="sparse_vector",
                param={"metric_type": "BM25"},
                limit=limit * group_size * self.config.search_over_fetch_factor,
                expr=expr,
            )

            headings_sparse_req = AnnSearchRequest(
                data=[query_text],
                anns_field="headings_sparse_vector",
                param={"metric_type": "BM25"},
                limit=limit * group_size * self.config.search_over_fetch_factor,
                expr=expr,
            )

            # WeightedRanker normalizes each sub-request's scores to [0,1] and sums
            # weight * normalized_score. Putting a higher weight on headings_bm25
            # makes an exact heading-path match outrank a pure body match.
            # Ref: https://milvus.io/docs/weighted-ranker.md
            hits = self.corpus_collection.hybrid_search(
                reqs=[dense_req, sparse_req, headings_sparse_req],
                rerank=WeightedRanker(
                    self.config.dense_weight,
                    self.config.chunk_bm25_weight,
                    self.config.heading_bm25_weight,
                ),
                limit=limit * group_size,
                output_fields=output_fields,
            )

            # hybrid_search returns results for the single query as hits[0]
            output.append(
                [
                    MilvusSearchHit(
                        id=hit.id,
                        distance=hit.distance,
                        doc_id=hit.entity.get("doc_id"),
                        file_name=hit.entity.get("file_name"),
                        chunk_text=hit.entity.get("chunk_text"),
                        parent_headings=json.loads(hit.entity.get("parent_headings")),
                        page_num=hit.entity.get("page_num"),
                    )
                    for hit in hits[0]
                ]
            )

        return output

    def delete_embeddings_for_doc_ids(self, docs: list[DocumentIDTopicPair]) -> MutationResult:
        """
        Delete embeddings for multiple documents.
        """
        if len(docs) == 0:
            return {"deleted_count": 0}

        quoted_ids = ",".join([f'"{doc.doc_id}"' for doc in docs])

        expr = f"doc_id in [{quoted_ids}]"

        result = self.corpus_collection.delete(expr)

        self.corpus_collection.flush()

        return result
