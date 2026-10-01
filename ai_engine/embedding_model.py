"""
=========================================================
Embedding Model

This module loads the sentence transformer model and
converts text into dense vector embeddings.

These embeddings are later used for semantic search,
similarity comparison, and document retrieval.

This module is part of the AI Engine.
=========================================================
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from ai_engine.config import EMBEDDING_MODEL


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer.
    """

    def __init__(self):

        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of texts into embeddings.

        Parameters
        ----------
        texts : List[str]
            Input texts.

        Returns
        -------
        np.ndarray
            Embedding vectors.
        """

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )