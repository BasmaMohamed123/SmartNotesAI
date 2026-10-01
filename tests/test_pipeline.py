"""
SmartNotesAI - AI Pipeline Test

This script tests the current AI pipeline:

1. Document processing
2. Text preprocessing
3. Chunking
4. Embeddings
5. Document classification
6. Semantic search
7. File loading

Run from the project root:

    python tests/test_pipeline.py
"""

import sys
from pathlib import Path

# ---------------------------------------------------------
# Make sure Python can find the ai_engine package
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Import SmartNotes AI components
# ---------------------------------------------------------

from ai_engine.document_processor import DocumentProcessor
from ai_engine.search_engine import DocumentSearchEngine
from ai_engine.document_loader import DocumentLoader


# =========================================================
# Helper functions
# =========================================================

def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check(condition, success_message, error_message):
    if condition:
        print(f"✅ {success_message}")
    else:
        print(f"❌ {error_message}")


# =========================================================
# TEST 1 - Document Processing
# =========================================================

def test_document_processing():

    print_section("TEST 1 - DOCUMENT PROCESSING")

    processor = DocumentProcessor()

    test_text = """
    Python is a popular programming language used for
    artificial intelligence, machine learning, data analysis,
    and software development.

    Machine learning allows computers to learn patterns from
    data and make predictions without being explicitly
    programmed for every possible situation.

    Python provides many useful libraries such as NumPy,
    Pandas, Scikit-learn, and PyTorch.
    """

    print("Processing test document...")

    document = processor.process_document(
        document_id=1,
        title="Python and Machine Learning",
        content=test_text,
        source="test",
        file_type="txt"
    )

    # -----------------------------------------------------
    # Check document
    # -----------------------------------------------------

    check(
        document is not None,
        "Document object created",
        "Document object was not created"
    )

    print(f"\nDocument:")
    print(f"  ID: {document.id}")
    print(f"  Title: {document.title}")
    print(f"  Category: {document.category}")
    print(f"  Classification score: "
          f"{document.classification_score:.4f}")
    print(f"  Number of chunks: {len(document.chunks)}")

    # -----------------------------------------------------
    # Check classification
    # -----------------------------------------------------

    check(
        bool(document.category),
        f"Classification completed → {document.category}",
        "Classification failed"
    )

    # -----------------------------------------------------
    # Check chunks
    # -----------------------------------------------------

    check(
        len(document.chunks) > 0,
        f"Chunking completed → {len(document.chunks)} chunks",
        "No chunks were created"
    )

    # -----------------------------------------------------
    # Check embeddings
    # -----------------------------------------------------

    if document.chunks:

        first_chunk = document.chunks[0]

        print("\nFirst chunk:")
        print(f"  Chunk ID: {first_chunk.id}")
        print(f"  Document ID: {first_chunk.document_id}")
        print(f"  Text: {first_chunk.text[:150]}...")

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
# TEST 2 - Semantic Search
# =========================================================

def test_semantic_search():

    print_section("TEST 2 - SEMANTIC SEARCH")

    engine = DocumentSearchEngine()

    # -----------------------------------------------------
    # Add documents
    # -----------------------------------------------------

    documents = [

        (
            "Python Programming",
            """
            Python is a high-level programming language.
            It is widely used in software development,
            artificial intelligence, machine learning,
            automation, and data analysis.
            """
        ),

        (
            "Machine Learning",
            """
            Machine learning is a branch of artificial
            intelligence where computers learn patterns
            from data and use those patterns to make
            predictions or decisions.
            """
        ),

        (
            "Travel Guide",
            """
            Egypt is a popular travel destination.
            Cairo contains many historical places and
            tourists can visit museums, pyramids, and
            other cultural attractions.
            """
        )
    ]

    print("Adding documents...")

    for title, content in documents:

        document = engine.add_document(
            title=title,
            content=content,
            source="test",
            file_type="note"
        )

        print(
            f"  Added: {document.title} "
            f"→ {document.category}"
        )

    # -----------------------------------------------------
    # Check number of documents
    # -----------------------------------------------------

    stored_documents = engine.get_documents()

    check(
        len(stored_documents) == 3,
        f"3 documents stored",
        f"Expected 3 documents, found {len(stored_documents)}"
    )

    # -----------------------------------------------------
    # Search
    # -----------------------------------------------------

    query = "How can I learn Python programming?"

    print(f"\nSearch query:")
    print(f"  {query}")

    results = engine.search(
        query=query,
        top_k=5
    )

    # -----------------------------------------------------
    # Check results
    # -----------------------------------------------------

    check(
        len(results) > 0,
        f"Search returned {len(results)} results",
        "Search returned no results"
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\nSearch Results:")

    for index, result in enumerate(results, start=1):

        print(f"\n#{index}")

        print(f"  Document: {result['title']}")
        print(f"  Category: {result['category']}")
        print(f"  Similarity: {result['similarity']}")
        print(f"  Chunk ID: {result['chunk_id']}")

        print(
            f"  Text: "
            f"{result['text'][:120]}..."
        )

    # -----------------------------------------------------
    # Check ranking
    # -----------------------------------------------------

    if results:

        top_result = results[0]

        print("\nTop result:")
        print(f"  {top_result['title']}")

        check(
            top_result["similarity"] >= 0,
            "Similarity score calculated correctly",
            "Invalid similarity score"
        )

    return results


# =========================================================
# TEST 3 - Document Loader
# =========================================================

def test_document_loader():

    print_section("TEST 3 - DOCUMENT LOADER")

    loader = DocumentLoader()

    test_file = (
        PROJECT_ROOT
        / "ai_engine"
        / "readers"
        / "test_note.txt"
    )

    print(f"Testing file:")
    print(f"  {test_file}")

    if not test_file.exists():

        print("⚠️ test_note.txt was not found.")
        print("Skipping file loader test.")

        return None

    try:

        text = loader.load(str(test_file))

        check(
            isinstance(text, str) and len(text.strip()) > 0,
            "TXT file loaded successfully",
            "TXT file was empty or could not be loaded"
        )

        print("\nExtracted text:")
        print(text[:300])

    except Exception as error:

        print("❌ File loading failed")
        print(f"Error: {error}")

        return None

    return text


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n")
    print("=" * 60)
    print("        SmartNotesAI - AI Pipeline Test")
    print("=" * 60)

    print("\nThis test checks:")
    print("  1. Document Processing")
    print("  2. Chunking")
    print("  3. Embeddings")
    print("  4. Classification")
    print("  5. Semantic Search")
    print("  6. Document Loading")

    try:

        # -------------------------------------------------
        # Run tests
        # -------------------------------------------------

        test_document_processing()

        test_semantic_search()

        test_document_loader()

        # -------------------------------------------------
        # Final result
        # -------------------------------------------------

        print_section("TEST FINISHED")

        print("🎉 SmartNotesAI pipeline test completed.")

        print("\nIf you see:")
        print("  ✅ Document processing")
        print("  ✅ Chunks")
        print("  ✅ Embeddings")
        print("  ✅ Classification")
        print("  ✅ Semantic search")
        print("  ✅ File loading")

        print("\nThen the current AI pipeline is working.")

    except Exception as error:

        print_section("TEST FAILED")

        print("❌ An error occurred:")
        print(error)

        print("\nWe will use the error message to identify")
        print("which part of the AI pipeline needs fixing.")


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    main()