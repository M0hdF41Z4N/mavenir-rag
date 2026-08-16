import pytest
from pymilvus import utility

from api.client.milvus_client import MilvusClient


@pytest.fixture
def milvus_client():
    temp_collection = "test_collection"

    dummy_config = {
        "host": "localhost",
        "port": "19530",
        "collection_name": temp_collection,  # <-- USE THIS DIRECTLY
        "vector_dim": 128,
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "M": 16,
        "ef_construction": 200,
    }

    client = MilvusClient(config_dict=dummy_config)

    yield client

    # Cleanup
    # This deletes the collection so next test run starts clean
    if utility.has_collection(temp_collection):
        utility.drop_collection(temp_collection)
        print(f"Temporary collection '{temp_collection}' deleted.")


def test_milvus_insert_and_search(milvus_client):
    """
    Test basic Milvus operations: insert and search.
    """
    milvus = milvus_client

    # Dummy embeddings
    vectors = [[1.0] + [0.0] * (milvus.vector_dim - 1), [0.2] * milvus.vector_dim]
    chunk_texts = ["dummy_text_1", "dummy_text_2"]
    file_names = ["dummy1.pdf", "dummy2.pdf"]

    # Insert embeddings
    inserted_ids = milvus.insert(vectors, file_names, chunk_texts)

    # Search for the first vector
    query = [[1.0] + [0.0] * (milvus.vector_dim - 1)]

    results = milvus.search(query, top_k=2)

    print("Search results:", results)

    # Assertions
    # Search should return a list.
    assert isinstance(results, list)
    # Because you sent 1 query vector, Milvus returns results per query.
    assert len(results) == 1
    # You asked top_k=2, so there should be 2 nearest neighbors.
    assert len(results[0]) == 2
    # The closest vector should be ID 1 because it's identical to the query.
    assert results[0][0]["id"] == inserted_ids[0]  # closest vector is the first inserted
    # distance must be non-negative
    assert results[0][0]["distance"] >= 0
