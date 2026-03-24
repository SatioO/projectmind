from typing import Literal, Dict, Union
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings

from config.settings import settings


EmbeddingProvider = Literal["openai", "huggingface"]


class Embedding:
    def __init__(self) -> None:
        self._cache: Dict[str,
                          Union[OpenAIEmbeddings, FastEmbedEmbeddings]] = {}

    def get_embedding_model(
        self,
        provider: EmbeddingProvider,
    ) -> Union[OpenAIEmbeddings, FastEmbedEmbeddings]:
        """
        Returns a cached embedding model instance per provider.
        """

        if provider in self._cache:
            return self._cache[provider]

        if provider == "openai":
            model = OpenAIEmbeddings(
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
                api_key=settings.openai_api_key,
            )

        elif provider == "huggingface":
            model = FastEmbedEmbeddings(
                model_name=settings.embedding_model
            )

        self._cache[provider] = model
        return model


embedding = Embedding()
