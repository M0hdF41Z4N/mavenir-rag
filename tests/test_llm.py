from api.client.llm_client import LLMClient


def test_chat_from_config():
    llm = LLMClient(config_path="config.yaml")
    response = llm.chat("Say hello in one short sentence.")
    assert isinstance(response, str)
    assert len(response.strip()) > 0


def test_embedding_from_config():
    llm = LLMClient(config_path="config.yaml")
    embedding = llm.embed("Testing embeddings from config-driven model.")
    assert isinstance(embedding, list)
    assert len(embedding) > 100
