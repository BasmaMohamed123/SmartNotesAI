
"""
=========================================================
SmartNotesAI - Classification Edge Cases Test

This test checks the document classifier with more
realistic and ambiguous documents.

The test includes:

1. Short documents
2. Mixed-topic documents
3. Meeting + technical content
4. Business + finance content
5. Programming + machine learning
6. Travel + personal content
7. Health + personal content
8. Course + programming content

The goal is to check how the zero-shot classifier behaves
when a document belongs to more than one possible category.
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
# HELPER
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
        "name": "Short Python Note",
        "text": "Python functions, variables, loops, and classes."
    },

    {
        "name": "Short Finance Note",
        "text": "Budget, savings, expenses, investments, and income."
    },

    {
        "name": "ML Project Meeting",
        "text": """
        Meeting with the AI team about our machine learning
        project. We discussed Python implementation, model
        training, dataset preparation, and the tasks that
        each team member should complete next week.
        """
    },

    {
        "name": "Business Finance Plan",
        "text": """
        Our company needs a financial plan for the next year.
        We discussed revenue, expenses, investment, pricing,
        customers, and business growth.
        """
    },

    {
        "name": "Python ML Project",
        "text": """
        We are developing a machine learning model using
        Python. The project includes data preprocessing,
        feature engineering, model training, and prediction.
        """
    },

    {
        "name": "Travel Personal Note",
        "text": """
        I am planning a personal trip to Cairo next month.
        I want to visit the pyramids, museums, historical
        places, and restaurants.
        """
    },

    {
        "name": "Healthy Lifestyle",
        "text": """
        My personal goal is to improve my health by exercising
        regularly, eating healthy food, sleeping better, and
        drinking enough water.
        """
    },

    {
        "name": "Python Programming Course",
        "text": """
        This course teaches Python programming, variables,
        functions, loops, object-oriented programming, and
        software development.
        """
    },

    {
        "name": "Cybersecurity Training",
        "text": """
        The training course covers cybersecurity fundamentals,
        network security, firewalls, encryption, passwords,
        malware, and vulnerability assessment.
        """
    },

    {
        "name": "Research Project",
        "text": """
        This research project evaluates different machine
        learning algorithms using experimental results,
        statistical analysis, and performance metrics.
        """
    }
]


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 60)
    print("SmartNotesAI - Classification Edge Cases Test")
    print("=" * 60)

    # -----------------------------------------------------
    # Load classifier
    # -----------------------------------------------------

    print()
    print("Loading classification model...")

    classifier = DocumentClassifier()

    print("Classification model loaded successfully.")


    # -----------------------------------------------------
    # Run tests
    # -----------------------------------------------------

    print_section("EDGE CASE CLASSIFICATION TESTS")

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
                    "OK: Strong classification score."
                )

            elif score >= 0.30:

                print(
                    "WARNING: Moderate classification score."
                )

            else:

                print(
                    "WARNING: Low classification score."
                )

        except Exception as error:

            print()
            print("ERROR: Classification failed.")
            print(f"Error: {error}")

            results.append(
                {
                    "name": document["name"],
                    "category": "ERROR",
                    "score": 0.0
                }
            )


    # =====================================================
    # SUMMARY
    # =====================================================

    print_section("EDGE CASE SUMMARY")

    successful = 0

    for result in results:

        if result["category"] != "ERROR":

            successful += 1

        print(
            f"{result['name']:<30}"
            f" -> {result['category']:<18}"
            f" Score: {result['score']:.4f}"
        )


    # =====================================================
    # FINAL RESULT
    # =====================================================

    print_section("FINAL RESULT")

    print(
        f"Documents classified successfully: "
        f"{successful}/{len(test_documents)}"
    )

    if successful == len(test_documents):

        print(
            "SUCCESS: The classifier handled all edge cases "
            "without runtime errors."
        )

    else:

        print(
            "WARNING: Some documents could not be classified."
        )

    print()
    print(
        "Classification Edge Cases Test Finished."
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()
