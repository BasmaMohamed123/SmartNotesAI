\# SmartNotesAI



An AI-powered personal knowledge management application that allows users to upload documents, process their content, classify them, and search through them using natural language and semantic similarity.



\## Features



\* Upload and process \*\*TXT, PDF, and DOCX\*\* documents

\* Extract and preprocess document text

\* Split documents into manageable chunks

\* Automatically classify uploaded documents

\* Generate semantic embeddings using \*\*Sentence Transformers\*\*

\* Perform natural-language semantic search

\* Retrieve the most relevant document sections

\* Keyword extraction

\* Local document storage and processing

\* REST API for the AI engine

\* Automated tests for core components and end-to-end workflows



\## AI \& NLP Pipeline



SmartNotesAI follows a document-processing and semantic-search pipeline:



```text

Document Upload

&#x20;     ↓

Text Extraction

&#x20;     ↓

Text Preprocessing

&#x20;     ↓

Document Chunking

&#x20;     ↓

Document Classification

&#x20;     ↓

Semantic Embeddings

&#x20;     ↓

Similarity Search

&#x20;     ↓

Relevant Results

```



\## Technologies



\* \*\*Python\*\*

\* \*\*PyTorch\*\*

\* \*\*Sentence Transformers\*\*

\* \*\*Hugging Face Transformers\*\*

\* \*\*Scikit-learn\*\*

\* \*\*FastAPI\*\*

\* \*\*Pandas\*\*

\* \*\*PyPDF\*\*

\* \*\*python-docx\*\*

\* \*\*Jupyter Notebook\*\*



\## Project Structure



```text

SmartNotesAI/

│

├── ai\_engine/

│   ├── document\_loader.py

│   ├── document\_processor.py

│   ├── chunking.py

│   ├── classification.py

│   ├── embedding\_model.py

│   ├── search\_engine.py

│   ├── keyword\_extraction.py

│   └── readers/

│

├── app/

│   └── app.py

│

├── notebooks/

│   └── 01\_Document\_Search.ipynb

│

├── tests/

│   ├── test\_classification.py

│   ├── test\_pipeline.py

│   ├── test\_end\_to\_end.py

│   └── ...

│

├── .gitignore

├── README.md

└── requirements.txt

```



\## How It Works



1\. The user uploads a document in TXT, PDF, or DOCX format.

2\. The application extracts the document text.

3\. The text is cleaned and divided into smaller chunks.

4\. The document can be classified automatically.

5\. Text chunks are converted into numerical embeddings.

6\. A semantic search query is converted into an embedding.

7\. The system compares the query with stored document embeddings.

8\. The most relevant document sections are returned to the user.



\## Running the Project



\### 1. Clone the repository



```bash

git clone https://github.com/BasmaMohamed123/SmartNotesAI.git

cd SmartNotesAI

```



\### 2. Create a virtual environment



```bash

python -m venv venv

```



\### 3. Activate the environment



Windows:



```bash

venv\\Scripts\\activate

```



\### 4. Install dependencies



```bash

pip install -r requirements.txt

```



\### 5. Run the application



```bash

python app/app.py

```



\## Testing



The project includes automated tests covering document processing, classification, semantic search, and end-to-end workflows.



Run the tests with:



```bash

pytest

```



\## Project Goal



SmartNotesAI was developed to explore practical applications of \*\*Artificial Intelligence, Natural Language Processing, document processing, and semantic search\*\* by combining multiple AI components into a single knowledge-management application.



