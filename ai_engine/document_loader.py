"""
=========================================================
Document Loader

This module selects the appropriate reader based on
the uploaded file type.

Supported formats:
- TXT
- PDF
- DOCX

It returns the extracted text regardless of the
original file format.

This module is part of the AI Engine.
=========================================================
"""

from pathlib import Path

from ai_engine.readers.txt_reader import TxtReader
from ai_engine.readers.pdf_reader import PDFReader
from ai_engine.readers.docx_reader import DocxReader


class DocumentLoader:
    """
    Loads documents using the appropriate reader.
    """

    def __init__(self):

        self.readers = {

            ".txt": TxtReader(),

            ".pdf": PDFReader(),

            ".docx": DocxReader()
        }

    def load(self, file_path: str) -> str:
        """
        Read a document and return its text.

        Parameters
        ----------
        file_path : str

        Returns
        -------
        str
            Extracted text.
        """

        extension = Path(file_path).suffix.lower()

        if extension not in self.readers:

            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        reader = self.readers[extension]

        return reader.read(file_path)