"""
=========================================================
Data Models

This file contains the core data structures used by
the AI Engine.

Document:
    Represents a complete user document.

Chunk:
    Represents a small part of a document.
    Each chunk has its own embedding for semantic search.
=========================================================
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np


@dataclass
class Chunk:
    """
    Represents a single chunk of a document.
    """

    id: int
    document_id: int
    text: str
    embedding: Optional[np.ndarray] = None


@dataclass
class Document:
    """
    Represents a complete document.
    """

    id: int

    title: str

    content: str

    # AI Classification
    category: str = "Unknown"

    classification_score: float = 0.0

    # Document Chunks
    chunks: List[Chunk] = field(default_factory=list)

    # Metadata
    source: str = "manual"

    file_type: str = "note"

    created_at: datetime = field(default_factory=datetime.now)

    def __repr__(self):

        return (
            f"Document("
            f"id={self.id}, "
            f"title='{self.title}', "
            f"category='{self.category}', "
            f"score={self.classification_score:.2f}, "
            f"chunks={len(self.chunks)})"
        )