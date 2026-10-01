"""
=========================================================
SmartNotesAI - Document Classification Test

This test checks whether the document classification
model assigns reasonable categories to different types
of documents.

Test categories:
1. Programming
2. Machine Learning / Research
3. Travel
4. Recipe
5. Course / Education
6. Business
7. Health
8. Finance
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

from ai_engine.classification import DocumentClassifier


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def print_section(title):

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# =========================================================
# TEST DOCUMENTS
# =========================================================

test_documents = [

    {
        "name": "Python Programming",
        "text": """
        Python is a programming language used to build
        software applications. Developers use Python,
        functions, classes, variables, libraries, and
        frameworks to create different types of software.
        """
    },

    {
        "name": "Machine Learning",
        "text": """
        Machine learning is a field of artificial
        intelligence where models learn patterns from
        training data and make predictions. Supervised
        learning, neural networks, classification, and
        regression are common machine learning techniques.
        """
    },

    {
        "name": "Travel Guide",
        "text": """
        Cairo is a popular tourist destination in Egypt.
        Visitors can explore the pyramids, museums,
        historical sites, ancient temples, and the Nile.
        """
    },

    {
        "name": "Cooking Recipe",
        "text": """
        To prepare pasta, boil water and add salt.
        Cook the pasta until it becomes tender, then
        prepare tomato sauce with garlic, onions,
        tomatoes, and herbs.
        """
    },

    {
        "name": "Cybersecurity Course",
        "text": """
        This course teaches cybersecurity fundamentals,
        network security, encryption, firewalls,
        authentication, malware protection, and
        vulnerability assessment.
        """
    },

    {
        "name": "Business Plan",
        "text": """
        A business plan describes the company's goals,
        target market, marketing strategy, competitors,
        revenue model, costs, and financial projections.
        """
    },

    {
        "name": "Health Information",
        "text": """
        Regular exercise, healthy nutrition, sufficient
        sleep, and drinking enough water are important
        parts of maintaining a healthy lifestyle.
        """
    },

    {
        "name": "Finance",
        "text": """
        Personal finance includes budgeting, saving,
        expenses, investments, income, interest rates,
        and managing financial goals.
        """
    }
]


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 60)
    print("SmartNotesAI - Document Classification Test")
    print("=" * 60)

    # -----------------------------------------------------
    # Create classifier
    # -----------------------------------------------------

    print()
    print("Loading classification model...")

    classifier = DocumentClassifier()

    print("Classification model loaded successfully.")


    # -----------------------------------------------------
    # Run classification tests
    # -----------------------------------------------------

    print_section("CLASSIFICATION TESTS")

    results = []

    for index, document in enumerate(
        test_documents,
        start=1
    ):

        print()
        print("-" * 60)

        print(f"Test #{index}")
        print(f"Document: {document['name']}")

        try:

            result = classifier.classify(
                document["text"]
            )

            category = result["category"]
            score = result["score"]

            print()
            print(f"Category: {category}")
            print(f"Score: {score:.4f}")

            results.append(
                {
                    "name": document["name"],
                    "category": category,
                    "score": score
                }
            )

            if score >= 0.50:

                print(
                    "OK: Classification confidence "
                    "is reasonable."
                )

            else:

                print(
                    "WARNING: Classification confidence "
                    "is relatively low."
                )

        except Exception as error:

            print()
            print("ERROR: Classification failed.")
            print(f"Error: {error}")

            results.append(
                {
                    "name": document["name"],
                    "category": "ERROR",
                    "score": 0
                }
            )


    # =====================================================
    # SUMMARY
    # =====================================================

    print_section("CLASSIFICATION SUMMARY")

    successful = 0

    for result in results:

        if result["category"] != "ERROR":
            successful += 1

        print(
            f"{result['name']:<25} "
            f"-> {result['category']:<15} "
            f"Score: {result['score']:.4f}"
        )


    print()
    print(
        f"Successfully classified: "
        f"{successful}/{len(test_documents)}"
    )


    # =====================================================
    # FINAL RESULT
    # =====================================================

    print_section("FINAL RESULT")

    if successful == len(test_documents):

        print(
            "SUCCESS: All documents were classified."
        )

    else:

        print(
            "WARNING: Some documents could not be classified."
        )


    print()
    print(
        "Document Classification Test Finished."
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()