"""
about.py
=============================================================================
DocuMind - About & Architecture Tab Component
Renders:
  - About this app & Core Value Proposition
  - High-Level Multimodal RAG Architecture
  - End-to-End Processing & Retrieval Flow
  - Step-by-Step "How to Use" Guide
  - Support & Help with Contact Information
All built strictly with native Streamlit elements (NO raw HTML/CSS).
=============================================================================
"""

import streamlit as st

def render_about_tab():
    """Renders the comprehensive About & Architecture view."""
    
    st.header("֎ About DocuMind")
    st.caption("Next-Generation Multimodal Document Intelligence & Grounded Question-Answering")
    
    st.info(
        "**DocuMind** bridges the critical gap in traditional document search by treating "
        "**Text, Complex Tables, and Visual Figures/Charts** as first-class citizens. "
        "Built on the cutting-edge RAG architecture from Google Gemini and Jina AI, "
        "it delivers grounded, verifiable answers with pinpoint source citations."
    )

    # Key Highlights Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Vector Embedding", "Jina v4", "2048 Dimensions")
    with col2:
        st.metric("Visual Engine", "Gemini Vision", "Multimodal Analysis")
    with col3:
        st.metric("Retrieval Index", "FAISS FlatIP", "Exact Cosine Match")
    with col4:
        st.metric("Generation Model", "Gemini 3.6 Flash", "Zero Hallucination")

    st.divider()

    # SECTION 1: SYSTEM ARCHITECTURE
    st.subheader("🏛️ System Architecture")
    st.write(
        "DocuMind operates on a 7-stage pipeline engineered for fidelity, accuracy, and depth. "
        "Below is the layer-by-layer architectural breakdown:"
    )

    with st.expander("📌 Layer 1: Universal Document Ingestion & Parsing", expanded=True):
        st.write(
            "- **Engine:** Docling / High-Fidelity PDF parser with native structure discovery.\n"
            "- **Capability:** Distinguishes text bodies, section headings, tabular structures, "
            "headers, footers, and visual boundaries.\n"
            "- **Output:** Normalized document objects linked with exact page numbers (`page_no`) and bounding boxes."
        )

    with st.expander("👁️ Layer 2: Gemini Multimodal Visual Understanding", expanded=True):
        st.write(
            "- **Engine:** Google Gemini Multimodal Vision API.\n"
            "- **Capability:** Automatically isolates graphics, flowcharts, architectural diagrams, and data plots. "
            "Filters candidates using resolution heuristics (`width >= 150px`, `height >= 80px`).\n"
            "- **Visual Reasoning Prompt:** Extracts chart titles, units, legends, axis categories, "
            "numerical trends, dates, and annotations into rich, searchable visual descriptions.\n"
            "- **Resilience:** Multi-tier exponential backoff and model fallbacks across Gemini Flash models."
        )

    with st.expander("🧩 Layer 3: Canonical Representation & Structure-Aware Chunking", expanded=True):
        st.write(
            "- **Canonical Elements:** Texts, tabular Markdown grids, and Gemini visual descriptions are aligned in true reading sequence.\n"
            "- **Section Anchoring:** Section headings are preserved and prepended to chunks as context anchors (`title: {section} | text: {content}`).\n"
            "- **Intact Tabular & Visual Chunks:** Tables and diagram descriptions are never split across arbitrary character boundaries.\n"
            "- **Boundary Optimization:** Running text is sentence-segmented and packed into target sizes (1,800 characters, max 3,200 characters)."
        )

    with st.expander("⚡ Layer 4: Jina v4 High-Dimensional Embeddings & FAISS Index", expanded=True):
        st.write(
            "- **Embedding Model:** `jina-embeddings-v4` (2048 dimensions).\n"
            "- **Task Separation:** Asymmetric retrieval using task `retrieval.passage` for document chunks and task `retrieval.query` for user questions.\n"
            "- **Batch Engine:** Safe rate-limited batching (50 chunks/batch) adhering to Jina RPM and TPM limits.\n"
            "- **Vector Database:** `faiss.IndexFlatIP` performing fast inner-product search on L2-normalized vectors (exact Cosine Similarity)."
        )

    with st.expander("🎯 Layer 5: Grounded Answer Generation & Citation Engine", expanded=True):
        st.write(
            "- **Synthesis Model:** Google Gemini 3.6 Flash.\n"
            "- **Grounding Framework:** Enforces strict boundary rules — no external hallucinations, exact numerical preservation, "
            "and multi-source comparative ranking.\n"
            "- **Citations:** Explicit page number references, table tags, and figure identifiers alongside every answer."
        )

    st.divider()

    # SECTION 2: PROCESSING FLOW
    st.subheader("🔄 End-to-End Processing & Retrieval Flow")
    
    flow_col1, flow_col2 = st.columns(2)
    
    with flow_col1:
        st.write("#### Ingestion & Indexing Pipeline")
        st.markdown(
            """
            1. **Upload PDF:** User provides document via drag-and-drop.
            2. **Deconstruction:** Text paragraphs, tables, and pictures are extracted.
            3. **Visual Comprehension:** Visuals are captioned by Gemini Vision.
            4. **Canonical Assembly:** All components ordered into unified elements.
            5. **Structure Chunking:** Sections, intact tables & visuals are chunked.
            6. **Jina Embedding:** Batched embeddings generated at 2048 dimensions.
            7. **FAISS Indexing:** Normalized vectors loaded into in-memory FAISS store.
            """
        )

    with flow_col2:
        st.write("#### Query & Retrieval Pipeline")
        st.markdown(
            """
            1. **User Query:** Entered via chat input or direct word check.
            2. **Query Embedding:** Transformed into 2048-dim vector via Jina `retrieval.query`.
            3. **Vector Search:** FAISS computes top-k cosine similarity matches.
            4. **Context Assembly:** Ranked sources assembled with metadata & page numbers.
            5. **Grounded Generation:** Gemini synthesizes accurate answer with citations.
            6. **Markdown Inspection:** User can click to inspect full source chunks.
            """
        )

    st.divider()

    # SECTION 3: HOW TO USE
    st.subheader("📖 How to Use DocuMind")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            **Step 1: Upload Your Document**
            - Switch to **Document Assistant** tab.
            - Upload any complex PDF (financial reports, research papers, manuals, contracts).
            - Click **Process Document** and follow the live progress bar.

            **Step 2: Instant Exact Word Search**
            - Use the **Direct Word Check** at the top of the tab.
            - Type any specific keyword, term, or entity name to pull exact matching chunks and pages instantly.
            """
        )
    with col_b:
        st.markdown(
            """
            **Step 3: Interactive Q&A Chat**
            - Ask questions about text, tables, or charts in the chat box.
            - DocuMind cites page numbers, tables, and figures for each answer.
            - Click **Inspect All Retrieved Chunks** to verify raw evidence.

            **Step 4: Verify in Scrollable PDF Viewer**
            - Use the built-in PDF viewer on the right side to inspect the exact pages referenced in the answer.
            """
        )

    st.divider()

    st.divider()

    # SECTION 4: API LIMITS & HOW TO USE YOUR OWN KEYS
    st.subheader("🔑 API Key Limits & Using Your Own Keys")
    st.warning(
        "**Safety Note:** If the default system API keys reach their rate limits or quotas during high traffic, "
        "you can seamlessly provide your own personal API keys to continue processing documents without interruption."
    )

    key_guide_col1, key_guide_col2 = st.columns(2)
    with key_guide_col1:
        st.write("#### 1️⃣ Google Gemini API Key (Vision & Synthesis)")
        st.markdown(
            """
            Google Gemini provides free API tiers with high rate limits for multimodal analysis and generation:
            - **How to get:** Visit [Google AI Studio](https://aistudio.google.com/app/apikey).
            - **Sign in** with your Google account.
            - Click **Create API key** and copy your key.
            - Paste your key in the **Sidebar -> 🔑 API Credentials -> Gemini API Key** field.
            """
        )
        st.link_button("🌐 Get Gemini API Key (Free)", "https://aistudio.google.com/app/apikey")

    with key_guide_col2:
        st.write("#### 2️⃣ Jina AI API Key (2048-dim Embeddings)")
        st.markdown(
            """
            Jina AI offers 1,000,000 free tokens upon signup for state-of-the-art document embeddings:
            - **How to get:** Visit [Jina AI](https://jina.ai/).
            - Sign up / Log in with your email or GitHub.
            - Go to the API Keys dashboard and copy your token.
            - Paste your token in the **Sidebar -> 🔑 API Credentials -> Jina API Key** field.
            """
        )
        st.link_button("🌐 Get Jina API Key (Free 1M Tokens)", "https://jina.ai/")

    st.divider()

    # SECTION 5: SUPPORT, DEVELOPER & CONTACT
    st.subheader("🤝 Support & Developer Contact")
    
    sup_col1, sup_col2 = st.columns(2)
    with sup_col1:
        st.write("#### 👨‍💻 Developer Profile")
        st.write("- **Developer:** **Gopinath S**")
        st.write("- **Email:** [gopinath.sara1n@gmail.com](mailto:gopinath.sara1n@gmail.com)")
        st.write("- **LinkedIn:** [linkedin.com/in/gopinaths](https://www.linkedin.com/in/gopinaths)")
        st.write("- **GitHub:** [github.com/gopinath-sara1n](https://github.com/gopinath-sara1n)")
        
        dev_btn_col1, dev_btn_col2 = st.columns(2)
        with dev_btn_col1:
            st.link_button("💼 LinkedIn Profile", "https://www.linkedin.com/in/gopinaths", use_container_width=True)
        with dev_btn_col2:
            st.link_button("🐙 GitHub Profile", "https://github.com/gopinath-sara1n", use_container_width=True)

    with sup_col2:
        st.write("#### 🛠️ Troubleshooting & Usage Tips")
        st.markdown(
            """
            - **Rate Limit Warnings (429):** If Jina or Gemini limits are hit, DocuMind will automatically retry with exponential backoff. You can also paste your own keys in the sidebar.
            - **Large PDFs:** For PDFs with many charts, Gemini Vision takes a few seconds per figure to extract full details.
            - **Document Switching:** Click **Clear Document & Reset** in the sidebar to process another document.
            - **Framework Documentation:** Built using [Streamlit](https://docs.streamlit.io/).
            """
        )

    st.success("DocuMind is engineered by Gopinath S with enterprise-grade precision!")
