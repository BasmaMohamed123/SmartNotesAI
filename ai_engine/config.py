
"""
=========================================================
AI Engine Configuration

This file contains all configurable constants used
throughout the AI Engine.

Changing values here automatically affects all modules.
=========================================================
"""

# =========================================================
# Embedding Model
# =========================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =========================================================
# Chunking
# =========================================================

CHUNK_SIZE = 800

CHUNK_OVERLAP = 150


# =========================================================
# Semantic Search
# =========================================================

TOP_K = 5

SIMILARITY_THRESHOLD = 0.30


# =========================================================
# Document Classification
# =========================================================

DOCUMENT_CATEGORIES = [
    "Programming",
    "Machine Learning",
    "Research",
    "Course",
    "Meeting",
    "Travel",
    "Recipe",
    "Cybersecurity",
    "Business",
    "Health",
    "Finance",
    "Personal"
]
