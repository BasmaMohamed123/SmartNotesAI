"""
=========================================================
SmartNotesAI - Semantic Search Quality Test

This test checks whether semantic search can understand
the meaning of a query and return the most relevant
document.

Test topics:
1. Python Programming
2. Machine Learning
3. Travel
4. Cooking
5. Cybersecurity
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
# ADD TEST DOCUMENTS
# =========================================================

def add_test_documents(engine):

    documents = [

        {
            "title": "Python Programming",
            "content": """
            Python is a high-level programming language used
            for software development, automation, data analysis,
            artificial intelligence, and web development.
            Python has a simple syntax and many useful libraries.
            """
        },

        {
            "title": "Machine Learning",
            "content": """
            Machine learning allows computers to learn patterns
            from data. Algorithms can be trained to make
            predictions and decisions without being explicitly
            programmed for every situation. Neural networks are
            widely used in modern machine learning systems.
            """
        },

        {
            "title": "Travel in Egypt",
            "content": """
            Egypt is a popular travel destination with many
            historical and cultural attractions. Visitors can
            explore Cairo, the pyramids, museums, ancient
            temples, and the Nile River.
            """
        },

        {
            "title": "Cooking Pasta",
            "content": """
            Pasta can be prepared by boiling it in salted water.
            Different sauces such as tomato sauce, cream sauce,
            and pesto can be added. Cooking time depends on the
            type of pasta.
            """
        },

        {
            "title": "Cybersecurity Basics",
            "content": """
            Cybersecurity protects computers, networks, and data
            from unauthorized access and attacks. Strong
            passwords, software updates, encryption, and
            firewalls can help improve security.
            """
        }
    ]

    print_section("ADDING TEST DOCUMENTS")

    for document in documents:

        result = engine.add_document(
            title=document["title"],
            content=document["content"],
            source="semantic_quality_test",
            file_type="note"
        )

        print(
            f"Added: {result.title} -> {result.category}"
        )

    print()

    check(
        len(engine.get_documents()) == 5,
        "5 test documents stored",
        f"Expected 5 documents, found "
        f"{len(engine.get_documents())}"
    )


# =========================================================
# RUN ONE SEARCH TEST
# =========================================================

def run_search_test(engine, query, expected_title):

    print()
    print("-" * 60)

    print("Query:")
    print(f"  {query}")

    print()
    print("Expected top result:")
    print(f"  {expected_title}")

    try:

        results = engine.search(
            query=query,
            top_k=5
        )

    except Exception as error:

        print()
        print("ERROR: Search failed.")
        print(f"Error: {error}")

        return False

    # -----------------------------------------------------
    # Check results
    # -----------------------------------------------------

    if not results:

        print()
        print("ERROR: No search results returned.")

        return False

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print()
    print("Results:")

    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"  #{index} "
            f"{result['title']} "
            f"(similarity: {result['similarity']})"
        )

    # -----------------------------------------------------
    # Check top result
    # -----------------------------------------------------

    top_result = results[0]

    print()
    print(
        f"Top result: "
        f"{top_result['title']}"
    )

    print(
        f"Similarity: "
        f"{top_result['similarity']}"
    )

    if top_result["title"] == expected_title:

        print(
            "OK: Correct document ranked first."
        )

        return True

    else:

        print(
            "ERROR: Incorrect document ranked first."
        )

        return False


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 60)
    print("SmartNotesAI - Semantic Search Quality Test")
    print("=" * 60)

    # -----------------------------------------------------
    # Create search engine
    # -----------------------------------------------------

    engine = DocumentSearchEngine()

    # -----------------------------------------------------
    # Add documents
    # -----------------------------------------------------

    add_test_documents(engine)

    # =====================================================
    # SEARCH TEST CASES
    # =====================================================

    test_cases = [

        (
            "How can I write software using Python?",
            "Python Programming"
        ),

        (
            "How do computers learn from examples and data?",
            "Machine Learning"
        ),

        (
            "What are the best historical places to visit?",
            "Travel in Egypt"
        ),

        (
            "How should I prepare pasta with sauce?",
            "Cooking Pasta"
        ),

        (
            "How can I protect my computer from hackers?",
            "Cybersecurity Basics"
        )
    ]

    print_section("SEMANTIC SEARCH TESTS")

    passed_tests = 0

    for query, expected_title in test_cases:

        success = run_search_test(
            engine,
            query,
            expected_title
        )

        if success:
            passed_tests += 1

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print_section("FINAL SUMMARY")

    total_tests = len(test_cases)

    print(
        f"Passed: "
        f"{passed_tests}/{total_tests}"
    )

    print(
        f"Failed: "
        f"{total_tests - passed_tests}/{total_tests}"
    )

    if passed_tests == total_tests:

        print()
        print(
            "SUCCESS: Semantic search passed all tests."
        )

        print(
            "The embedding-based search is correctly "
            "ranking relevant documents."
        )

    elif passed_tests >= 3:

        print()
        print(
            "PARTIAL SUCCESS: Semantic search is working,"
        )

        print(
            "but some queries need further investigation."
        )

    else:

        print()
        print(
            "WARNING: Semantic search needs investigation."
        )

    print()
    print(
        "Semantic Search Quality Test Finished."
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()