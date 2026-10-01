"""
=========================================================
Base Reader

Abstract base class for all document readers.

Every reader must implement the read() method
and return the extracted text as a string.

Examples:
    PDFReader
    DocxReader
    TxtReader
=========================================================
"""

from abc import ABC, abstractmethod


class BaseReader(ABC):
    """
    Base class for document readers.
    """

    @abstractmethod
    def read(self, file_path: str) -> str:
        """
        Read a document and return its text.

        Parameters
        ----------
        file_path : str
            Path to the document.

        Returns
        -------
        str
            Extracted text.
        """
        pass