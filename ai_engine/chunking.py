"""
=========================================================
Chunking Module

This module is responsible for splitting long documents
into semantic chunks before generating embeddings.

Instead of cutting text at random positions, it tries
to preserve the meaning by splitting at paragraphs,
sentences, and finally words if needed.

Each chunk will later be embedded and stored for
semantic search.

This module is part of the AI Engine.
=========================================================
"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_engine.config import CHUNK_SIZE, CHUNK_OVERLAP


class TextChunker:
    """
    Splits long documents into semantic chunks.
    """

    def __init__(self):

        self.splitter = RecursiveCharacterTextSplitter(

            chunk_size=CHUNK_SIZE,

            chunk_overlap=CHUNK_OVERLAP,

            separators=[
                "\n\n",      # Paragraphs
                "\n",        # New lines
                ". ",        # Sentences
                "? ",
                "! ",
                "; ",
                ", ",
                " ",         # Words
                ""           # Characters (last resort)
            ],

            keep_separator=True,

            length_function=len,

            is_separator_regex=False
        )

    def split_text(self, text: str) -> List[str]:
        """
        Split a document into semantic chunks.

        Parameters
        ----------
        text : str
            Input document.

        Returns
        -------
        List[str]
            List of chunks.
        """

        if not text or not text.strip():
            return []

        return self.splitter.split_text(text)