# 🧠 DocuMind — Intelligent Document Assistant

> **A Next-Generation Multimodal Document Intelligence & Grounded Question-Answering App built with Streamlit, Google Gemini Vision, Jina v4 Embeddings, and FAISS.**

---

## 🌟 Overview

Standard Retrieval-Augmented Generation (RAG) systems fail on complex documents because they strip away diagrams, charts, and tables, converting them into mangled, flat text.

**DocuMind** solves this limitation by introducing a **7-Stage Multimodal RAG Pipeline**:
1. **Universal Parsing:** Isolates text blocks, section headers, tables, and graphic elements.
2. **Gemini Vision Understanding:** Sends visual charts, diagrams, and figures directly to **Google Gemini Multimodal Vision** to generate deep, factual, searchable visual summaries.
3. **Canonical Assembly:** Preserves true reading order across text, structured markdown tables, and visual descriptions.
4. **Structure-Aware Chunking:** Keeps tables and visual descriptions intact as standalone units while segmenting text along sentence boundaries.
5. **High-Dimensional Embeddings:** Encodes chunks using **Jina v4 Embeddings (`jina-embeddings-v4`)** at **2048 dimensions** with asymmetric task separation (`retrieval.passage` vs. `retrieval.query`).
6. **FAISS Vector Database:** Performs exact Cosine Similarity matching via `faiss.IndexFlatIP` on L2-normalized vectors.
7. **Grounded Generation with Citations:** Employs **Google Gemini 3.6 Flash** under strict zero-hallucination prompts, returning exact answers with page citations, table references, and figure tags.

---

## 🚀 Key Features

- **Shiftable Dual-Tab Interface:**
  - **Tab 1: 📄 Document Assistant:**
    - **🔎 Direct Word & Keyword Check:** Scan all document chunks immediately for exact keywords, acronyms, or numbers.
    - **Live Progress Updates:** Step-by-step progress tracking during document ingestion.
    - **Document Intelligence Metrics:** Pages, Chunks, Tables, Visuals, Canonical Elements, and Indexing Duration.
    - **Interactive Chat Assistant:** Conversational memory (`st.session_state`), precise citations, and zero-hallucination responses.
    - **🔍 Markdown Chunk Inspector:** Inspect the raw retrieved chunks and similarity scores with a single click.
    - **📑 Scrollable PDF Viewer:** High-resolution page navigation and synchronized preview to verify answers in the source PDF.
    - **🔄 Clear Document:** Wipe session state with one click to process a new document.
  - **Tab 2: ℹ️ About & Architecture:**
    - Architectural breakdown, end-to-end processing flow, "How to Use" guide, and contact support.
- **Pure Streamlit Design:** 100% native Streamlit components (`st.status`, `st.metric`, `st.chat_message`, `st.expander`, `st.tabs`, etc.) with **no custom HTML/CSS**, ensuring seamless cross-platform deployment.

---

## 📁 Project Structure

```text
├── .env                  # API keys (GEMINI_API_KEY, JINA_API_KEY)
├── app.py                # Main Streamlit application
├── rag_pipeline.py       # Core 7-stage Multimodal RAG engine
├── about.py              # Tab 2: About, Architecture, Flow & Support
├── requirements.txt      # Python dependencies
├── packages.txt          # Linux packages for Streamlit Cloud
└── README.md             # Documentation & Setup Guide
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.10 to 3.13
- A Google Gemini API Key
- A Jina AI API Key

### 2. Clone or Navigate to Project
```bash
cd aps
```

### 3. Create & Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure API Keys
API keys can be placed in `.env` in the project root:
```env
GEMINI_API_KEY="your_gemini_api_key"
JINA_API_KEY="your_jina_api_key"
```
*(You can also configure or override keys directly in the app's sidebar).*

---

## 🏃 Running the Application

Launch the Streamlit app with:
```bash
streamlit run app.py
```

Once started, open your browser at `http://localhost:8501`.

---

## ☁️ Deployment (Streamlit Cloud)

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a **New App**.
3. Point to your repository and set Main file path to `app.py`.
4. In **Advanced Settings -> Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key"
   JINA_API_KEY = "your_jina_api_key"
   ```
5. Deploy! Streamlit Cloud will automatically install dependencies from `requirements.txt` and system packages from `packages.txt`.

---

## 🔑 Obtaining API Keys (Free Tiers Available)

If your default rate limit is reached, you can generate and plug in your own free personal API keys:
- **Google Gemini API Key:** [Google AI Studio (Get Free Gemini Key)](https://aistudio.google.com/app/apikey)
- **Jina AI API Key:** [Jina AI (Get Free 1M Token Key)](https://jina.ai/)

---

## 🤝 Support & Developer Contact

- **Developer:** **Gopinath S**
- **Email:** [gopinath.sara1n@gmail.com](mailto:gopinath.sara1n@gmail.com)
- **LinkedIn:** [linkedin.com/in/gopinaths](https://www.linkedin.com/in/gopinaths)
- **GitHub:** [github.com/gopinath-sara1n](https://github.com/gopinath-sara1n)
- **Project Documentation:** [docs.streamlit.io](https://docs.streamlit.io/)
