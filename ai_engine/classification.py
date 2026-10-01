
"""
=========================================================
Document Classifier

This module automatically predicts the category of a
document using Zero-Shot Classification.

Model:
    facebook/bart-large-mnli

No training dataset is required.

This module is part of the AI Engine.
=========================================================
"""

from transformers import pipeline

from ai_engine.config import DOCUMENT_CATEGORIES


class DocumentClassifier:
    """
    Zero-shot document classifier for SmartNotesAI.
    """

    def __init__(self):

        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

    # =====================================================
    # CLASSIFY DOCUMENT
    # =====================================================

    def classify(self, text: str) -> dict:
        """
        Predict the category of a document.

        Parameters
        ----------
        text : str
            Document text.

        Returns
        -------
        dict
            {
                "category": "...",
                "score": ...
            }
        """

        # -------------------------------------------------
        # Empty text
        # -------------------------------------------------

        if not text or not text.strip():

            return {
                "category": "Unknown",
                "score": 0.0
            }

        # -------------------------------------------------
        # Zero-shot classification
        # -------------------------------------------------

        result = self.classifier(
            text,
            candidate_labels=DOCUMENT_CATEGORIES,
            multi_label=False,
            hypothesis_template="This document is about {}."
        )

        # -------------------------------------------------
        # Best category
        # -------------------------------------------------

        category = result["labels"][0]

        score = float(result["scores"][0])

        return {
            "category": category,
            "score": score
        }
