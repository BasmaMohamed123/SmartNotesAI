"""
=========================================================
Semantic Search Engine

This module is responsible for:

1. Loading documents from text or files.
2. Processing documents.
3. Storing processed documents.
4. Performing semantic search.

This module is the main interface of the AI Engine.
=========================================================
"""

from pathlib import Path
from typing import List, Optional

from sklearn.metrics.pairwise import cosine_similarity

from ai_engine.config import TOP_K
from ai_engine.document_loader import DocumentLoader
from ai_engine.document_processor import DocumentProcessor
from ai_engine.models import Document


class DocumentSearchEngine:
    """
    Semantic Search Engine.
    """

    def __init__(self):

        self.processor = DocumentProcessor()

        self.loader = DocumentLoader()

        self.documents: List[Document] = []

        self.next_document_id = 1

    # =====================================================
    # Add document from plain text
    # =====================================================

    def add_document(
        self,
        title: str,
        content: str,
        source: str = "manual",
        file_type: str = "note"
    ) -> Document:
        """
        Process and store a text document.
        """

        document = self.processor.process_document(
            document_id=self.next_document_id,
            title=title,
            content=content,
            source=source,
            file_type=file_type
        )

        self.documents.append(document)

        self.next_document_id += 1

        return document

    # =====================================================
    # Add document from file
    # =====================================================

    def add_document_from_file(
        self,
        file_path: str,
        source: str = "upload"
    ) -> Document:
        """
        Load a document from a file, process it,
        and store it.
        """

        text = self.loader.load(file_path)

        title = Path(file_path).stem

        file_type = Path(file_path).suffix.lower().replace(".", "")

        return self.add_document(
            title=title,
            content=text,
            source=source,
            file_type=file_type
        )

    # =====================================================
    # Get all documents
    # =====================================================

    def get_documents(self) -> List[Document]:
        """
        Return all stored documents.
        """

        return self.documents

    # =====================================================
    # Semantic Search
    # =====================================================

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        category: Optional[str] = None
    ) -> List[dict]:
        """
        Perform semantic search across all document chunks.

        Parameters
        ----------
        query : str
            User search query.

        top_k : int
            Number of returned results.

        category : str, optional
            Filter search by document category.

        Returns
        -------
        List[dict]
            Ranked search results.
        """

        query_embedding = self.processor.embedding_model.encode([query])[0]

        results = []

        for document in self.documents:

            # Skip documents from other categories (optional)
            if category is not None:

                if document.category.lower() != category.lower():

                    continue

            for chunk in document.chunks:

                similarity = cosine_similarity(
                    [query_embedding],
                    [chunk.embedding]
                )[0][0]

                results.append({

                    "document_id": document.id,

                    "title": document.title,

                    "category": document.category,

                    "classification_score": round(
                        document.classification_score,
                        3
                    ),

                    "file_type": document.file_type,

                    "source": document.source,

                    "chunk_id": chunk.id,

                    "text": chunk.text,

                    "similarity": round(
                        float(similarity),
                        4
                    )
                })

        results.sort(
            key=lambda x: x["similarity"],
            reverse=True
        )

        return results[:top_k]