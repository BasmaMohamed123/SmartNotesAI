"""
=========================================================
Keyword Extraction

This module extracts important keywords and topics
from a document.

The extracted keywords can later be used for:
- Search
- Tags
- Recommendations
- Document metadata

This module is part of the AI Engine.
=========================================================
"""

from typing import List

from transformers import pipeline


class KeywordExtractor:
    """
    Extract important keywords from documents.
    """

    def __init__(self):

        self.extractor = pipeline(
            task="text2text-generation",
            model="google/flan-t5-base"
        )

    def extract(
        self,
        text: str,
        max_keywords: int = 10
    ) -> List[str]:
        """
        Extract important keywords from text.

        Parameters
        ----------
        text : str
            Document text.

        max_keywords : int
            Maximum number of keywords to return.

        Returns
        -------
        List[str]
            Extracted keywords.
        """

        if not text or not text.strip():
            return []

        prompt = f"""
Extract the {max_keywords} most important keywords or topics
from the following text.

Return only the keywords separated by commas.

Text:
{text}
"""

        result = self.extractor(
            prompt,
            max_new_tokens=100,
            do_sample=False
        )

        output = result[0]["generated_text"]

        keywords = [
            keyword.strip()
            for keyword in output.split(",")
            if keyword.strip()
        ]

        return keywords[:max_keywords]