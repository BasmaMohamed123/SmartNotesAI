
"""
=========================================================
SmartNotesAI - Real File Semantic Search Test

This test verifies the complete pipeline using real files:

1. Load TXT / PDF / DOCX
2. Extract text
3. Process the document
4. Generate chunks
5. Generate embeddings
6. Classify the document
7. Store the document
8. Perform semantic search
9. Return ranked results

This is the final local test before database integration.
=========================================================
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
# HELPERS
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
# MAIN
# =========================================================

def main():

    print()
    print("=" * 60)
    print("SmartNotesAI - Real File Semantic Search Test")
    print("=" * 60)

    # -----------------------------------------------------
    # Create search engine
    # -----------------------------------------------------

    print()
    print("Initializing search engine...")

    engine = DocumentSearchEngine()

    print("Search engine initialized successfully.")


    # =====================================================
    # TEST FILES
    # =====================================================

    test_files_dir = (
        PROJECT_ROOT
        / "tests"
        / "test_files"
    )

    txt_file = test_files_dir / "test_note.txt"
    pdf_file = test_files_dir / "test_note.pdf"
    docx_file = test_files_dir / "test_note.docx"


    files = [
        ("TXT", txt_file),
        ("PDF", pdf_file),
        ("DOCX", docx_file)
    ]


    # =====================================================
    # LOAD REAL FILES
    # =====================================================

    print_section("ADDING REAL FILES")

    added_documents = []

    for file_type, file_path in files:

        print()
        print("-" * 60)

        print(f"File type: {file_type}")
        print(f"File: {file_path}")

        # -------------------------------------------------
        # Check file
        # -------------------------------------------------

        if not file_path.exists():

            print(
                f"WARNING: {file_type} file not found."
            )

            continue

        # -------------------------------------------------
        # Add file
        # -------------------------------------------------

        try:

            document = engine.add_document_from_file(
                file_path=str(file_path),
                source="real_file_test"
            )

            added_documents.append(document)

            print()
            print(
                f"OK: {file_type} added successfully."
            )

            print(
                f"  ID: {document.id}"
            )

            print(
                f"  Title: {document.title}"
            )

            print(
                f"  File type: {document.file_type}"
            )

            print(
                f"  Category: {document.category}"
            )

            print(
                f"  Classification score: "
                f"{document.classification_score:.4f}"
            )

            print(
                f"  Chunks: {len(document.chunks)}"
            )

            # -------------------------------------------------
            # Validate document
            # -------------------------------------------------

            check(
                document.content is not None,
                f"{file_type} content loaded",
                f"{file_type} content is empty"
            )

            check(
                len(document.chunks) > 0,
                f"{file_type} chunks created",
                f"{file_type} has no chunks"
            )

            if document.chunks:

                embedding = document.chunks[0].embedding

                check(
                    embedding is not None,
                    f"{file_type} embedding generated",
                    f"{file_type} embedding missing"
                )

                if embedding is not None:

                    print(
                        f"  Embedding shape: "
                        f"{embedding.shape}"
                    )

        except Exception as error:

            print()
            print(
                f"ERROR: Failed to add {file_type}."
            )

            print(
                f"Error: {error}"
            )


    # =====================================================
    # CHECK STORED DOCUMENTS
    # =====================================================

    print_section("STORED DOCUMENTS")

    documents = engine.get_documents()

    print(
        f"Documents stored: {len(documents)}"
    )

    check(
        len(documents) == len(added_documents),
        "All successfully added files are stored",
        "Stored document count does not match"
    )


    for document in documents:

        print()
        print(
            f"ID: {document.id}"
        )

        print(
            f"Title: {document.title}"
        )

        print(
            f"Type: {document.file_type}"
        )

        print(
            f"Category: {document.category}"
        )


    # =====================================================
    # SEMANTIC SEARCH
    # =====================================================

    print_section("SEMANTIC SEARCH")

    queries = [

        "What is Python and how is it used?",

        "How do computers learn from data?",

        "What is artificial intelligence?"
    ]


    for query in queries:

        print()
        print("-" * 60)

        print(
            f"Query: {query}"
        )

        try:

            results = engine.search(
                query=query,
                top_k=3
            )

            check(
                len(results) > 0,
                f"Search returned {len(results)} results",
                "Search returned no results"
            )

            if not results:
                continue

            print()
            print("Ranked Results:")

            for index, result in enumerate(
                results,
                start=1
            ):

                print()
                print(
                    f"#{index}"
                )

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
                    f"{result['text'][:180]}..."
                )

        except Exception as error:

            print()
            print(
                "ERROR: Search failed."
            )

            print(
                f"Error: {error}"
            )


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print_section("FINAL SUMMARY")

    print(
        f"Files successfully added: "
        f"{len(added_documents)}/3"
    )

    print(
        f"Documents stored in engine: "
        f"{len(engine.get_documents())}"
    )

    if len(added_documents) == 3:

        print()
        print(
            "SUCCESS: TXT, PDF, and DOCX were successfully "
            "loaded, processed, embedded, classified, "
            "stored, and searched."
        )

        print()
        print(
            "The local AI pipeline is ready for "
            "database integration."
        )

    else:

        print()
        print(
            "WARNING: Not all real files were processed."
        )

    print()
    print(
        "Real File Semantic Search Test Finished."
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()
