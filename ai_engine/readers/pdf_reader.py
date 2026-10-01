"""
=========================================================
PDF Reader

Reads PDF documents and extracts their text.

Each page is read sequentially and combined into
a single string.

This reader is part of the AI Engine.
=========================================================
"""

from pypdf import PdfReader

from ai_engine.readers.base_reader import BaseReader


class PDFReader(BaseReader):
    """
    Reader for PDF documents.
    """

    def read(self, file_path: str) -> str:
        """
        Read a PDF file and extract its text.
        """

        reader = PdfReader(file_path)

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:

                pages.append(text)

        return "\n".join(pages)