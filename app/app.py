
"""
=========================================================
SmartNotesAI - AI Demo Interface

A simple AI-powered personal knowledge base.

Features:
1. Upload TXT / PDF / DOCX files
2. Upload and store images
3. Extract document text
4. Automatically classify documents
5. Generate semantic embeddings
6. Store processed documents locally
7. Search using natural language
8. Display relevant documents
9. Download uploaded files
=========================================================
"""

import sys
from pathlib import Path

import streamlit as st


# =========================================================
# Project Path
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# AI Engine Import
# =========================================================

from ai_engine.search_engine import DocumentSearchEngine


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="SmartNotesAI",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# Local Storage
# =========================================================

STORAGE_DIR = PROJECT_ROOT / "storage"

DOCUMENTS_DIR = STORAGE_DIR / "documents"
IMAGES_DIR = STORAGE_DIR / "images"

DOCUMENTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Title
# =========================================================

st.title("🧠 SmartNotesAI")

st.markdown(
    """
    ### Your AI-Powered Personal Knowledge Base

    Save your documents and search through them using
    natural language.

    The AI Engine automatically:

    - 📄 Extracts text from documents
    - 🧩 Splits documents into chunks
    - 🧠 Generates semantic embeddings
    - 🏷️ Classifies documents automatically
    - 🔎 Performs semantic search
    """
)


# =========================================================
# Initialize AI Search Engine
# =========================================================

@st.cache_resource
def create_search_engine():
    return DocumentSearchEngine()


search_engine = create_search_engine()


# =========================================================
# Session State
# =========================================================

if "processed_file_names" not in st.session_state:
    st.session_state.processed_file_names = set()


# =========================================================
# Sidebar
# =========================================================

st.sidebar.title("🧠 SmartNotesAI")

st.sidebar.markdown("---")

st.sidebar.subheader("AI Engine")

st.sidebar.write(
    "Local AI processing is enabled."
)

st.sidebar.write(
    "No database is required for this AI demo."
)

st.sidebar.markdown("---")

documents = search_engine.get_documents()

st.sidebar.metric(
    "📚 Documents",
    len(documents)
)

total_chunks = sum(
    len(document.chunks)
    for document in documents
)

st.sidebar.metric(
    "🧩 Chunks",
    total_chunks
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "AI Pipeline:\n"
    "Document → Text Extraction → Cleaning → "
    "Chunking → Embeddings → Classification → Search"
)


# =========================================================
# Main Tabs
# =========================================================

tab_upload, tab_search, tab_documents, tab_images = st.tabs(
    [
        "📤 Upload",
        "🔎 Search",
        "📚 Documents",
        "🖼️ Images"
    ]
)


# =========================================================
# TAB 1 - UPLOAD DOCUMENTS
# =========================================================

with tab_upload:

    st.header("📤 Upload Documents")

    st.write(
        "Upload TXT, PDF, or DOCX files. "
        "The AI Engine will automatically process them."
    )

    uploaded_files = st.file_uploader(
        "Choose document files",
        type=[
            "txt",
            "pdf",
            "docx"
        ],
        accept_multiple_files=True,
        key="document_uploader"
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            file_name = uploaded_file.name

            # -------------------------------------------------
            # Avoid processing the same file during the session
            # -------------------------------------------------

            if file_name in st.session_state.processed_file_names:
                continue

            file_path = DOCUMENTS_DIR / file_name

            # -------------------------------------------------
            # Save original file
            # -------------------------------------------------

            try:

                with open(
                    file_path,
                    "wb"
                ) as file:

                    file.write(
                        uploaded_file.getbuffer()
                    )

            except Exception as error:

                st.error(
                    f"❌ Could not save {file_name}"
                )

                st.exception(error)

                continue

            # -------------------------------------------------
            # Process with AI Engine
            # -------------------------------------------------

            with st.spinner(
                f"🧠 AI is processing {file_name}..."
            ):

                try:

                    document = (
                        search_engine
                        .add_document_from_file(
                            str(file_path),
                            source="streamlit_upload"
                        )
                    )

                    st.session_state.processed_file_names.add(
                        file_name
                    )

                    st.success(
                        f"✅ {file_name} processed successfully!"
                    )

                    # -------------------------------------------------
                    # AI Results
                    # -------------------------------------------------

                    st.write(
                        "### 🧠 AI Processing Result"
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Category",
                            document.category
                        )

                    with col2:

                        st.metric(
                            "Confidence",
                            f"{document.classification_score:.2f}"
                        )

                    with col3:

                        st.metric(
                            "Chunks",
                            len(document.chunks)
                        )

                    st.markdown("---")

                    st.write(
                        f"**Document:** "
                        f"{document.title}"
                    )

                    st.write(
                        f"**File type:** "
                        f"{document.file_type.upper()}"
                    )

                    st.write(
                        f"**Category:** "
                        f"{document.category}"
                    )

                    st.write(
                        f"**Classification score:** "
                        f"{document.classification_score:.4f}"
                    )

                    # -------------------------------------------------
                    # Embedding Information
                    # -------------------------------------------------

                    if document.chunks:

                        embedding_dimension = len(
                            document.chunks[0].embedding
                        )

                        st.write(
                            f"**Embedding dimension:** "
                            f"{embedding_dimension}"
                        )

                        st.write(
                            f"**Number of chunks:** "
                            f"{len(document.chunks)}"
                        )

                except Exception as error:

                    st.error(
                        f"❌ Failed to process {file_name}"
                    )

                    st.exception(error)


    # =====================================================
    # Upload Images
    # =====================================================

    st.markdown("---")

    st.header("🖼️ Save Important Images")

    st.write(
        "Upload images that you want to keep accessible "
        "inside your SmartNotesAI project."
    )

    uploaded_images = st.file_uploader(
        "Choose image files",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ],
        accept_multiple_files=True,
        key="image_uploader"
    )

    if uploaded_images:

        for image_index, uploaded_image in enumerate(
            uploaded_images
        ):

            image_name = uploaded_image.name

            image_path = IMAGES_DIR / image_name

            try:

                with open(
                    image_path,
                    "wb"
                ) as file:

                    file.write(
                        uploaded_image.getbuffer()
                    )

                st.success(
                    f"🖼️ {image_name} saved successfully."
                )

            except Exception as error:

                st.error(
                    f"❌ Could not save {image_name}"
                )

                st.exception(error)


# =========================================================
# TAB 2 - SEMANTIC SEARCH
# =========================================================

with tab_search:

    st.header("🔎 Semantic Search")

    st.write(
        """
        Search your documents using natural language.

        You do not need to use the exact words that appear
        inside the document.
        """
    )

    query = st.text_input(
        "What are you looking for?",
        placeholder=(
            "Example: How do computers learn from data?"
        ),
        key="search_query"
    )

    search_button = st.button(
        "🔍 Search",
        type="primary",
        use_container_width=True,
        key="semantic_search_button"
    )

    if search_button:

        if not query.strip():

            st.warning(
                "Please enter a search query."
            )

        elif not search_engine.get_documents():

            st.warning(
                "Please upload at least one document first."
            )

        else:

            with st.spinner(
                "🧠 Searching semantically..."
            ):

                try:

                    results = search_engine.search(
                        query=query,
                        top_k=5
                    )

                    if not results:

                        st.info(
                            "No relevant documents found."
                        )

                    else:

                        st.success(
                            f"Found {len(results)} "
                            f"relevant results."
                        )

                        st.markdown("---")

                        # -------------------------------------------------
                        # Search Results
                        # -------------------------------------------------

                        for result_index, result in enumerate(
                            results,
                            start=1
                        ):

                            st.subheader(
                                f"#{result_index} — "
                                f"{result['title']}"
                            )

                            col1, col2, col3, col4 = st.columns(4)

                            with col1:

                                st.write(
                                    f"📄 **Type**\n"
                                    f"{result['file_type'].upper()}"
                                )

                            with col2:

                                st.write(
                                    f"🏷️ **Category**\n"
                                    f"{result['category']}"
                                )

                            with col3:

                                st.write(
                                    f"🎯 **Similarity**\n"
                                    f"{result['similarity']:.4f}"
                                )

                            with col4:

                                st.write(
                                    f"🧠 **Classification**\n"
                                    f"{result['classification_score']:.4f}"
                                )

                            st.write(
                                "**Relevant content:**"
                            )

                            st.info(
                                result["text"]
                            )

                            # -------------------------------------------------
                            # Find original file safely
                            # -------------------------------------------------

                            original_file = None

                            # First try using document_id
                            document_id = result.get(
                                "document_id"
                            )

                            if document_id is not None:

                                for stored_document in search_engine.get_documents():

                                    if stored_document.id == document_id:

                                        candidate = (
                                            DOCUMENTS_DIR
                                            / (
                                                stored_document.title
                                                + "."
                                                + stored_document.file_type
                                            )
                                        )

                                        if candidate.exists():

                                            original_file = candidate

                                        break

                            # -------------------------------------------------
                            # Fallback: search by title and file type
                            # -------------------------------------------------

                            if original_file is None:

                                candidate = (
                                    DOCUMENTS_DIR
                                    / (
                                        result["title"]
                                        + "."
                                        + result["file_type"]
                                    )
                                )

                                if candidate.exists():

                                    original_file = candidate

                            # -------------------------------------------------
                            # Download button
                            # -------------------------------------------------

                            if original_file is not None:

                                try:

                                    with open(
                                        original_file,
                                        "rb"
                                    ) as file:

                                        st.download_button(
                                            label="⬇️ Download Original File",
                                            data=file.read(),
                                            file_name=original_file.name,
                                            key=(
                                                f"search_download_"
                                                f"{result_index}_"
                                                f"{result.get('document_id', 'none')}_"
                                                f"{result['file_type']}"
                                            )
                                        )

                                except Exception as error:

                                    st.warning(
                                        "Could not prepare "
                                        "the file for download."
                                    )

                                    st.exception(error)

                            st.markdown("---")

                except Exception as error:

                    st.error(
                        "❌ Search failed."
                    )

                    st.exception(error)


# =========================================================
# TAB 3 - STORED DOCUMENTS
# =========================================================

with tab_documents:

    st.header("📚 Your Documents")

    documents = search_engine.get_documents()

    if not documents:

        st.info(
            "No documents have been uploaded yet."
        )

    else:

        st.write(
            f"You currently have "
            f"**{len(documents)} documents**."
        )

        st.markdown("---")

        # -------------------------------------------------
        # enumerate gives every widget a unique index
        # -------------------------------------------------

        for document_index, document in enumerate(
            documents
        ):

            # Unique identifier for all widgets
            document_key = (
                f"{document_index}_"
                f"{document.id}_"
                f"{document.file_type}"
            )

            with st.expander(
                f"📄 {document.title} "
                f"({document.file_type.upper()})"
            ):

                # -------------------------------------------------
                # Document Information
                # -------------------------------------------------

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.write(
                        f"**Category:** "
                        f"{document.category}"
                    )

                with col2:

                    st.write(
                        f"**Classification:** "
                        f"{document.classification_score:.4f}"
                    )

                with col3:

                    st.write(
                        f"**Chunks:** "
                        f"{len(document.chunks)}"
                    )

                st.markdown("---")

                # -------------------------------------------------
                # Extracted Text
                # -------------------------------------------------

                st.write(
                    "### 📄 Extracted Text"
                )

                st.text_area(
                    "Document content",
                    document.content,
                    height=180,
                    key=f"content_{document_key}"
                )

                # -------------------------------------------------
                # Embedding Information
                # -------------------------------------------------

                if document.chunks:

                    embedding_dimension = len(
                        document.chunks[0].embedding
                    )

                    st.write(
                        f"🧠 Embedding dimension: "
                        f"**{embedding_dimension}**"
                    )

                    st.write(
                        f"🧩 Number of chunks: "
                        f"**{len(document.chunks)}**"
                    )

                # -------------------------------------------------
                # Original File Download
                # -------------------------------------------------

                original_file = (
                    DOCUMENTS_DIR
                    / (
                        document.title
                        + "."
                        + document.file_type
                    )
                )

                if original_file.exists():

                    try:

                        with open(
                            original_file,
                            "rb"
                        ) as file:

                            st.download_button(
                                label="⬇️ Download Original File",
                                data=file.read(),
                                file_name=original_file.name,
                                key=(
                                    f"document_download_"
                                    f"{document_key}"
                                )
                            )

                    except Exception as error:

                        st.warning(
                            "Could not prepare "
                            "the document for download."
                        )

                        st.exception(error)


# =========================================================
# TAB 4 - IMAGES
# =========================================================

with tab_images:

    st.header("🖼️ Saved Images")

    image_files = []

    for extension in [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.webp"
    ]:

        image_files.extend(
            IMAGES_DIR.glob(extension)
        )

    # Remove duplicates and sort
    image_files = sorted(
        set(image_files),
        key=lambda path: path.name.lower()
    )

    if not image_files:

        st.info(
            "No images have been uploaded yet."
        )

    else:

        st.write(
            f"Saved images: "
            f"**{len(image_files)}**"
        )

        st.markdown("---")

        for image_index, image_path in enumerate(
            image_files
        ):

            col1, col2 = st.columns(
                [2, 1]
            )

            with col1:

                st.image(
                    str(image_path),
                    caption=image_path.name,
                    width=350
                )

            with col2:

                st.write(
                    f"**File:** "
                    f"{image_path.name}"
                )

                try:

                    with open(
                        image_path,
                        "rb"
                    ) as file:

                        st.download_button(
                            label="⬇️ Download Image",
                            data=file.read(),
                            file_name=image_path.name,
                            key=(
                                f"image_download_"
                                f"{image_index}_"
                                f"{image_path.name}"
                            )
                        )

                except Exception as error:

                    st.error(
                        "Could not load image."
                    )

                    st.exception(error)

            st.markdown("---")


# =========================================================
# AI PIPELINE STATUS
# =========================================================

st.markdown("---")

st.header("⚙️ AI Pipeline Status")

documents = search_engine.get_documents()

total_chunks = sum(
    len(document.chunks)
    for document in documents
)

embedding_dimension = 0

if documents:

    for document in documents:

        if document.chunks:

            embedding_dimension = len(
                document.chunks[0].embedding
            )

            break


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "📚 Documents",
        len(documents)
    )

with col2:

    st.metric(
        "🧩 Chunks",
        total_chunks
    )

with col3:

    st.metric(
        "🧠 Embedding Dimension",
        embedding_dimension
    )

with col4:

    st.metric(
        "🔎 Search",
        "Semantic"
    )


# =========================================================
# Footer
# =========================================================

st.markdown("---")

st.caption(
    "SmartNotesAI — AI Engine Demo | "
    "Document Processing • Classification • "
    "Embeddings • Semantic Search"
)