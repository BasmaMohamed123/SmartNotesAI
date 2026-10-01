"""
SmartNotesAI - End-to-End AI Pipeline Test

Tests the complete pipeline:

File
  ↓
DocumentLoader
  ↓
Text Extraction
  ↓
DocumentProcessor
  ↓
Cleaning
  ↓
Chunking
  ↓
Embeddings
  ↓
Classification
  ↓
DocumentSearchEngine
  ↓
Semantic Search
"""

import sys
from pathlib import Path


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# IMPORT
# =========================================================

from ai_engine.search_engine import DocumentSearchEngine


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def print_section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def check(condition, success_message, error_message):
    if condition:
        print(f"OK: {success_message}")
    else:
        print(f"ERROR: {error_message}")


# =========================================================
# TEST ONE FILE
# =========================================================

def test_file(engine, file_path, file_type):

    print_section(f"TEST - {file_type}")

    print(f"File: {file_path}")

    # -----------------------------------------------------
    # Check file exists
    # -----------------------------------------------------

    if not file_path.exists():
        print(f"WARNING: {file_type} file not found.")
        print(f"Expected: {file_path}")
        return None

    # -----------------------------------------------------
    # Add document from file
    # -----------------------------------------------------

    try:
        document = engine.add_document_from_file(
            file_path=str(file_path),
            source="test_upload"
        )

    except Exception as error:
        print(f"ERROR: Could not process {file_type}.")
        print(f"Error: {error}")
        return None

    # =====================================================
    # DOCUMENT INFORMATION
    # =====================================================

    print()
    print("Document Information:")

    print(f"  ID: {document.id}")
    print(f"  Title: {document.title}")
    print(f"  File type: {document.file_type}")
    print(f"  Source: {document.source}")
    print(f"  Category: {document.category}")
    print(
        f"  Classification score: "
        f"{document.classification_score:.4f}"
    )

    check(
        document is not None,
        "Document created",
        "Document was not created"
    )

    # =====================================================
    # CONTENT
    # =====================================================

    check(
        isinstance(document.content, str),
        "Document content is a string",
        "Document content is not a string"
    )

    check(
        len(document.content.strip()) > 0,
        "Document text extracted",
        "Document content is empty"
    )

    print()
    print("Extracted Text Preview:")
    print("-" * 50)
    print(document.content[:300])
    print("-" * 50)

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    check(
        bool(document.category),
        f"Classification completed -> {document.category}",
        "Classification failed"
    )

    # =====================================================
    # CHUNKS
    # =====================================================

    print()
    print(f"Number of chunks: {len(document.chunks)}")

    check(
        len(document.chunks) > 0,
        f"Chunks created -> {len(document.chunks)}",
        "No chunks were created"
    )

    # =====================================================
    # EMBEDDINGS
    # =====================================================

    if document.chunks:

        first_chunk = document.chunks[0]

        print()
        print("First Chunk:")

        print(f"  Chunk ID: {first_chunk.id}")
        print(f"  Document ID: {first_chunk.document_id}")

        print(
            f"  Text: "
            f"{first_chunk.text[:150]}..."
        )

        check(
            first_chunk.embedding is not None,
            "Embedding generated",
            "Embedding was not generated"
        )

        if first_chunk.embedding is not None:

            print(
                f"  Embedding shape: "
                f"{first_chunk.embedding.shape}"
            )

            check(
                len(first_chunk.embedding) == 384,
                "Embedding dimension = 384",
                f"Unexpected embedding dimension: "
                f"{len(first_chunk.embedding)}"
            )

    return document


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 60)
    print("SmartNotesAI - End-to-End AI Pipeline Test")
    print("=" * 60)

    # -----------------------------------------------------
    # Create search engine
    # -----------------------------------------------------

    engine = DocumentSearchEngine()

    # -----------------------------------------------------
    # Test files directory
    # -----------------------------------------------------

    test_files_dir = (
        PROJECT_ROOT
        / "tests"
        / "test_files"
    )

    txt_file = test_files_dir / "test_note.txt"
    pdf_file = test_files_dir / "test_note.pdf"
    docx_file = test_files_dir / "test_note.docx"

    # =====================================================
    # TEST TXT
    # =====================================================

    txt_document = test_file(
        engine,
        txt_file,
        "TXT"
    )

    # =====================================================
    # TEST PDF
    # =====================================================

    pdf_document = test_file(
        engine,
        pdf_file,
        "PDF"
    )

    # =====================================================
    # TEST DOCX
    # =====================================================

    docx_document = test_file(
        engine,
        docx_file,
        "DOCX"
    )

    # =====================================================
    # SEMANTIC SEARCH TEST
    # =====================================================

    print_section("SEMANTIC SEARCH TEST")

    query = "How does machine learning work?"

    print()
    print(f"Query: {query}")

    try:

        results = engine.search(
            query=query,
            top_k=5
        )

        check(
            len(results) > 0,
            f"Search returned {len(results)} results",
            "Search returned no results"
        )

        print()
        print("Search Results:")

        for index, result in enumerate(
            results,
            start=1
        ):

            print()
            print(f"#{index}")

            print(
                f"  Document: "
                f"{result['title']}"
            )

            print(
                f"  File type: "
                f"{result['file_type']}"
            )

            print(
                f"  Category: "
                f"{result['category']}"
            )

            print(
                f"  Similarity: "
                f"{result['similarity']}"
            )

            print(
                f"  Text: "
                f"{result['text'][:150]}..."
            )

    except Exception as error:

        print("ERROR: Semantic search failed.")
        print(f"Error: {error}")

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print_section("FINAL SUMMARY")

    print(
        "TXT  : "
        + ("PASS" if txt_document else "FAILED")
    )

    print(
        "PDF  : "
        + ("PASS" if pdf_document else "FAILED")
    )

    print(
        "DOCX : "
        + ("PASS" if docx_document else "FAILED")
    )

    if len(engine.get_documents()) == 3:

        print(
            "SEARCH ENGINE : PASS "
            "(3 documents stored)"
        )

    else:

        print(
            "SEARCH ENGINE : FAILED"
        )

    print()
    print("End-to-End AI Pipeline Test Finished.")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()