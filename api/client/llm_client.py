# Thin wrapper around LiteLLM that normalizes chat, vision, and embedding calls to a
# single interface. Swapping providers (Ollama → OpenAI → Gemini) only requires a
# config change — call sites stay unchanged.
import base64

from litellm import completion, embedding

from api.config import LLMSettings
from api.models import ChatMessage


# Turn on debug logging
# litellm._turn_on_debug()
class LLMClient:
    """
    litellm_chat_model examples:
        - "ollama/gemma3:4b"
        - "openai/gpt-4o-mini"
        - "gemini/gemini-1.5-flash"

    litellm_embed_model examples:
        - "ollama/nomic-embed-text"
        - "openai/text-embedding-3-small"
    """

    def __init__(self, settings: LLMSettings) -> None:
        self.provider = settings.provider
        self.litellm_chat_model = settings.litellm_chat_model
        self.litellm_embed_model = settings.litellm_embed_model
        self.api_base = settings.ollama_api_base if settings.provider == "ollama" else None
        self.chat_temperature = settings.chat_temperature

        if not self.litellm_chat_model or not self.litellm_embed_model:
            raise ValueError("litellm_chat_model and litellm_embed_model must be set in config")

    def chat(self, messages: list[ChatMessage]) -> str:
        # Use LiteLLM Chat Completions API
        # Ref: https://docs.litellm.ai/docs/completion/usage#quick-start
        response = completion(
            model=self.litellm_chat_model,
            messages=messages,
            api_base=self.api_base,
            temperature=self.chat_temperature,
        )

        return response.choices[0].message.content

    def chat_with_image(self, prompt: str, image_bytes: bytes, image_media_type: str = "image/png") -> str:
        """Send a vision request with inline base64 image bytes."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{image_media_type};base64,{b64}"},
                    },
                ],
            }
        ]
        response = completion(model=self.litellm_chat_model, messages=messages, api_base=self.api_base)
        return response.choices[0].message.content

    def embed(self, text: str) -> list[float]:
        # Ref: https://docs.litellm.ai/docs/embedding/supported_embedding#quick-start
        resp = embedding(model=self.litellm_embed_model, input=text, api_base=self.api_base)
        return resp["data"][0]["embedding"]
