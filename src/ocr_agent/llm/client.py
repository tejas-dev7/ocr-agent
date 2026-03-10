"""LiteLLM wrapper for multi-provider LLM access."""

from ocr_agent.config import OCRConfig, get_config
from ocr_agent.llm.audit import ensure_audit_registered


class LLMClient:
    """Thin wrapper around LiteLLM for OpenAI, Claude, Ollama."""

    def __init__(self, config: OCRConfig | None = None):
        self.config = config or get_config()
        self.model = self.config.llm_model
        ensure_audit_registered()

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        metadata: dict | None = None,
        **kwargs,
    ) -> str:
        """Send chat completion and return content."""
        from litellm import completion

        model = model or self.model
        litellm_kwargs = {**kwargs}
        if metadata:
            litellm_kwargs["metadata"] = metadata
        response = completion(
            model=model,
            messages=messages,
            **litellm_kwargs,
        )
        return response.choices[0].message.content or ""

    def query_with_context(
        self,
        question: str,
        context: str,
        model: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """RAG-style Q&A: answer question given context."""
        messages = [
            {
                "role": "system",
                "content": "Answer the question based only on the provided context. If the context does not contain the answer, say so.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]
        return self.complete(messages, model=model, metadata=metadata)
