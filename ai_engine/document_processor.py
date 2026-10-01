"""
=========================================================
Document Processor

This module prepares documents before they are stored
or searched.

Pipeline:
1. Clean the text
2. Split it into chunks
3. Generate embeddings for each chunk
4. Classify the document
5. Build the final Document object

This module is part of the AI Engine.
=========================================================
"""

from ai_engine.models import Document, Chunk
from ai_engine.preprocessing import TextPreprocessor
from ai_engine.chunking import TextChunker
from ai_engine.embedding_model import EmbeddingModel
from ai_engine.classification import DocumentClassifier


class DocumentProcessor:
    """
    Processes documents into searchable chunks.
    """

    def __init__(self):

        self.chunker = TextChunker()

        self.embedding_model = EmbeddingModel()

        self.classifier = DocumentClassifier()

    def process_document(
        self,
        document_id: int,
        title: str,
        content: str,
        source: str = "manual",
        file_type: str = "note"
    ) -> Document:
        """
        Process a document and return a Document object
        containing embedded chunks.
        """

        # -------------------------------------------------
        # Step 1: Clean the text
        # -------------------------------------------------

        clean_text = TextPreprocessor.clean(content)

        # -------------------------------------------------
        # Step 2: Split into chunks
        # -------------------------------------------------

        chunk_texts = self.chunker.split_text(clean_text)

        # -------------------------------------------------
        # Step 3: Generate embeddings
        # -------------------------------------------------

        embeddings = self.embedding_model.encode(chunk_texts)

        # -------------------------------------------------
        # Step 4: Create Chunk objects
        # -------------------------------------------------

        chunks = []

        for index, (text, embedding) in enumerate(
            zip(chunk_texts, embeddings),
            start=1
        ):

            chunk = Chunk(
                id=index,
                document_id=document_id,
                text=text,
                embedding=embedding
            )

            chunks.append(chunk)

        # -------------------------------------------------
        # Step 5: Classify the document
        # -------------------------------------------------

        classification = self.classifier.classify(clean_text)

        category = classification["category"]

        classification_score = classification["score"]

        # -------------------------------------------------
        # Step 6: Build Document object
        # -------------------------------------------------

        document = Document(
            id=document_id,
            title=title,
            content=clean_text,
            category=category,
            classification_score=classification_score,
            chunks=chunks,
            source=source,
            file_type=file_type
        )

        return document