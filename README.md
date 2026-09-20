# ֎ DocuMind — Intelligent Document Assistant using RAG

**DocuMind** is a multimodal Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents and ask natural-language questions about their content.

It processes **text, tables, figures, and images**, performs semantic retrieval using vector embeddings, and generates **grounded answers with source references**.

### 🔗 Live Demo

**[🚀 Try DocuMind](https://intelligent-document-assistant-using-rag-eyrresdnxfmbjputjrytd.streamlit.app/)**

---

## Features

* 📄 PDF processing with **Docling**
* 📊 Table extraction and structure preservation
* 🖼️ Figure and image understanding using **Google Gemini**
* ✂️ Structure-aware document chunking
* 🔎 Semantic search using **Jina Embeddings v4**
* ⚡ Vector similarity search with **FAISS**
* 🔤 Exact keyword/phrase search
* 🤖 Grounded question answering with **Google Gemini**
* 📑 Page and source references
* 💬 Conversational Q&A
* 📖 Integrated PDF viewer for source verification

## Architecture

```text
                PDF Upload
                    │
                    ▼
        ┌──────────────────────┐
        │   Document Parsing   │
        │       Docling        │
        └──────────┬───────────┘
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
        Text     Tables   Figures
          │        │        │
          │        │        ▼
          │        │   Gemini Vision
          │        │        │
          └────────┼────────┘
                   ▼
          Canonical Document
                   │
                   ▼
        Structure-Aware Chunking
                   │
                   ▼
          Jina Embeddings v4
                   │
                   ▼
              FAISS Index
                   │
             User Question
                   │
                   ▼
          Semantic Retrieval
                   │
                   ▼
          Retrieved Context
                   │
                   ▼
           Gemini Generation
                   │
                   ▼
       Grounded Answer + Sources
```

## Tech Stack

| Component            | Technology            |
| -------------------- | --------------------- |
| UI                   | Streamlit             |
| PDF Processing       | Docling               |
| PDF Fallback         | pdfplumber, pypdfium2 |
| Visual Understanding | Google Gemini         |
| Embeddings           | Jina Embeddings v4    |
| Vector Search        | FAISS                 |
| Answer Generation    | Google Gemini         |
| Image Processing     | Pillow                |
| Data Processing      | NumPy, Pandas         |

## Project Structure

```text
├── app.py              # Streamlit application
├── rag_pipeline.py     # Document processing and RAG pipeline
├── about.py            # About & architecture interface
├── requirements.txt    # Python dependencies
├── packages.txt        # System dependencies
├── LICENSE
└── README.md
```

## How It Works

### 1. Upload

Upload a PDF through the Streamlit interface.

### 2. Process

DocuMind extracts:

* Text
* Tables
* Figures/images

Visual content is analysed using Gemini and converted into searchable descriptions.

### 3. Index

Document content is chunked and converted into **2048-dimensional Jina embeddings**. Normalized embeddings are stored in a **FAISS `IndexFlatIP`** index.

### 4. Retrieve

Questions are processed using semantic retrieval. Exact keyword search is also available for precise terms and phrases.

### 5. Generate

Retrieved document context is passed to Gemini with a grounding prompt to generate answers based on the available evidence.

### 6. Verify

Source/page information is provided with answers, with an integrated PDF viewer for verification.

## Installation

```bash
git clone https://github.com/gopinath-sara1n/Intelligent-Document-Assistant-using-RAG.git
cd Intelligent-Document-Assistant-using-RAG

python -m venv venv
venv\Scripts\activate        # Windows

pip install -r requirements.txt
pip install docling
```

## API Keys

DocuMind requires:

* **Google Gemini API Key** — visual analysis and answer generation
* **Jina AI API Key** — document and query embeddings

Configure them using environment variables:

```env
GEMINI_API_KEY=your_gemini_api_key
JINA_API_KEY=your_jina_api_key
```

API keys can also be provided through the application's sidebar.

> Never commit API keys to the repository.

## Run Locally

```bash
streamlit run app.py
```

## Example Questions

```text
What is the main objective of this document?

What does the graph on page 12 show?

According to Table 4, what was the value in 2024?

What are the key findings of the report?
```

## RAG Pipeline

```text
PDF
 ↓
Docling
 ↓
Text + Tables + Visuals
 ↓
Gemini Vision
 ↓
Chunking
 ↓
Jina Embeddings v4
 ↓
FAISS
 ↓
Semantic / Keyword Retrieval
 ↓
Gemini
 ↓
Grounded Answer + Sources
```

## Contact

**Gopinath S**

🔗 **LinkedIn:** [linkedin.com/in/gopinaths](https://www.linkedin.com/in/gopinaths/?utm_source=chatgpt.com)

🚀 **Live Application:** [DocuMind — Streamlit App](https://intelligent-document-assistant-using-rag-eyrresdnxfmbjputjrytd.streamlit.app/?utm_source=chatgpt.com)

## License

This project is licensed under the [MIT License](LICENSE).
