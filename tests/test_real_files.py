"""
SmartNotesAI - Real File Loading Test

Tests the actual file-loading pipeline for:
1. TXT
2. PDF
3. DOCX
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

from ai_engine.document_loader import DocumentLoader


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def print_section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def test_file(loader, file_path, file_type):
    print()
    print(f"Testing {file_type}:")
    print(f"File: {file_path}")

    # Check if file exists
    if not file_path.exists():
        print(f"WARNING: {file_type} file not found.")
        print(f"Expected location: {file_path}")
        return False

    # Try to load the file
    try:
        text = loader.load(str(file_path))

    except Exception as error:
        print(f"ERROR: {file_type} loading failed.")
        print(f"Error: {error}")
        return False

    # Check returned type
    if isinstance(text, str):
        print(f"OK: {file_type} returned a string.")
    else:
        print(f"ERROR: {file_type} did not return a string.")
        return False

    # Check extracted text
    if len(text.strip()) > 0:
        print(f"OK: {file_type} text extracted successfully.")
    else:
        print(f"ERROR: {file_type} returned empty text.")
        return False

    # Display preview
    print()
    print("Extracted text preview:")
    print("-" * 50)

    print(text.strip()[:500])

    print("-" * 50)

    return True


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 60)
    print("SmartNotesAI - Real File Loading Test")
    print("=" * 60)

    # Create DocumentLoader
    loader = DocumentLoader()

    # -----------------------------------------------------
    # Test files directory
    # -----------------------------------------------------

    test_files_dir = PROJECT_ROOT / "tests" / "test_files"

    txt_file = test_files_dir / "test_note.txt"
    pdf_file = test_files_dir / "test_note.pdf"
    docx_file = test_files_dir / "test_note.docx"

    # -----------------------------------------------------
    # TXT
    # -----------------------------------------------------

    print_section("TEST 1 - TXT")

    txt_result = test_file(
        loader,
        txt_file,
        "TXT"
    )

    # -----------------------------------------------------
    # PDF
    # -----------------------------------------------------

    print_section("TEST 2 - PDF")

    pdf_result = test_file(
        loader,
        pdf_file,
        "PDF"
    )

    # -----------------------------------------------------
    # DOCX
    # -----------------------------------------------------

    print_section("TEST 3 - DOCX")

    docx_result = test_file(
        loader,
        docx_file,
        "DOCX"
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print_section("TEST SUMMARY")

    if txt_result:
        print("TXT  : PASS")
    else:
        print("TXT  : FAILED / NOT FOUND")

    if pdf_result:
        print("PDF  : PASS")
    else:
        print("PDF  : FAILED / NOT FOUND")

    if docx_result:
        print("DOCX : PASS")
    else:
        print("DOCX : FAILED / NOT FOUND")

    print()
    print("Real file loading test finished.")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()