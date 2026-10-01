"""
=========================================================
TXT Reader

Reads plain text (.txt) files and returns their content.

This reader is used before document processing.
=========================================================
"""

from ai_engine.readers.base_reader import BaseReader


class TxtReader(BaseReader):
    """
    Reader for text files.
    """

    def read(self, file_path: str) -> str:
        """
        Read a TXT file.

        Parameters
        ----------
        file_path : str
            Path to the text file.

        Returns
        -------
        str
            File content.
        """

        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()