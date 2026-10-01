import re


class TextPreprocessor:
    """
    Cleans and normalizes text before generating embeddings.
    """

    @staticmethod
    def clean(text: str) -> str:
        """
        Clean input text by:
        - Removing extra spaces
        - Removing new lines
        - Removing tabs
        """

        if not isinstance(text, str):
            return ""

        text = text.replace("\n", " ")
        text = text.replace("\t", " ")

        text = re.sub(r"\s+", " ", text)

        return text.strip()