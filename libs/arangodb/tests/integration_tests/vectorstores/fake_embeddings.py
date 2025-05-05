"""Fake Embedding class for testing purposes."""

from typing import List

from langchain_core.embeddings import Embeddings

class FakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    def __init__(self, dim: int = 10):
        self.dim = dim
        # Attribute to store predefined texts for consistent query embedding
        self.test_texts: List[str] = [] 

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings based on index in the input list."""
        return [[float(i + 1)] * self.dim for i, _ in enumerate(texts)]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents."""
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings based on index of text in predefined list."""
        try:
            # Find index in the predefined test_texts list
            idx = self.test_texts.index(text)
            return [float(idx + 1)] * self.dim
        except (AttributeError, ValueError, IndexError):
            # Fallback for arbitrary query text or if test_texts not set/found
            # Use hash for some determinism, scaled to a small range
            return [float(hash(text) % 10)] * self.dim 

    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query."""
        return self.embed_query(text) 