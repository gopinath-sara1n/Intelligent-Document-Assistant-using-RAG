"""
app.py
=============================================================================
DocuMind (Intelligent Document Assistant)
Streamlit Application Entrypoint
Connecting:
  - rag_pipeline.py: High-fidelity Multimodal RAG Engine
  - about.py: Architecture, Flow, and Documentation
Strictly using native Streamlit elements (NO raw HTML/CSS).
=============================================================================
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
import pypdfium2 as pdfium

import rag_pipeline
from about import render_about_tab

# ---------------------------------------------------------------------------
# 1. PAGE SETUP & CONFIGURATION
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DocuMind - Intelligent Document Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# 2. SESSION STATE INITIALIZATION
# ---------------------------------------------------------------------------

if "doc_index" not in st.session_state:
    st.session_state.doc_index = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "temp_pdf_path" not in st.session_state:
    st.session_state.temp_pdf_path = None

if "current_view_page" not in st.session_state:
    st.session_state.current_view_page = 1

if "keyword_results" not in st.session_state:
    st.session_state.keyword_results = None

# ---------------------------------------------------------------------------
# 3. SIDEBAR: CREDENTIALS & CONTROLS
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Environment keys with fallback (from .env, os.environ, or st.secrets)
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    jina_key = os.environ.get("JINA_API_KEY", "")
    try:
        if not gemini_key and hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            gemini_key = st.secrets["GEMINI_API_KEY"]
            os.environ["GEMINI_API_KEY"] = gemini_key
        if not jina_key and hasattr(st, "secrets") and "JINA_API_KEY" in st.secrets:
            jina_key = st.secrets["JINA_API_KEY"]
            os.environ["JINA_API_KEY"] = jina_key
    except Exception:
        pass
    
    with st.expander("🔑 API Credentials", expanded=not (gemini_key and jina_key)):
        user_gemini_key = st.text_input(
            "Gemini API Key",
            value=gemini_key,
            type="password",
            help="Google Gemini API key for visual understanding and answer synthesis."
        )
        user_jina_key = st.text_input(
            "Jina API Key",
            value=jina_key,
            type="password",
            help="Jina API key for 2048-dimensional embeddings."
        )
        st.caption("ℹ️ *If rate limit is reached, enter your personal free API keys:*")
        st.markdown("- [Get Gemini API Key (Free)](https://aistudio.google.com/app/apikey)")
        st.markdown("- [Get Jina API Key (Free)](https://jina.ai/)")

        if user_gemini_key:
            gemini_key = user_gemini_key
            os.environ["GEMINI_API_KEY"] = user_gemini_key
        if user_jina_key:
            jina_key = user_jina_key
            os.environ["JINA_API_KEY"] = user_jina_key

    # Status indicators
    st.subheader("System Status")
    if gemini_key:
        st.success("Gemini API: Connected")
    else:
        st.error("Gemini API: Key Missing")
        
    if jina_key:
        st.success("Jina Embedding API: Connected")
    else:
        st.error("Jina Embedding API: Key Missing")

    st.divider()
    
    # RAG Settings
    st.subheader("Retrieval Settings")
    top_k = st.slider(
        "Top-K Chunks to Retrieve",
        min_value=2,
        max_value=10,
        value=5,
        help="Number of semantically relevant chunks passed to Gemini as evidence."
    )

    st.divider()

    # Document Management Action
    if st.session_state.doc_index is not None:
        st.subheader("Active Document")
        st.info(f"📄 {st.session_state.uploaded_file_name}")
        
        if st.button("🗑️ Clear Document & Reset", type="secondary", use_container_width=True):
            st.session_state.doc_index = None
            st.session_state.messages = []
            st.session_state.uploaded_file_name = None
            st.session_state.temp_pdf_path = None
            st.session_state.keyword_results = None
            st.session_state.current_view_page = 1
            st.rerun()

    st.divider()
    st.caption("👨‍💻 **Developed by Gopinath S**")
    sb_col_a, sb_col_b = st.columns(2)
    with sb_col_a:
        st.link_button("💼 LinkedIn", "https://www.linkedin.com/in/gopinaths", use_container_width=True)
    with sb_col_b:
        st.link_button("🐙 GitHub", "https://github.com/gopinath-sara1n", use_container_width=True)

# ---------------------------------------------------------------------------
# 4. MAIN HEADER
# ---------------------------------------------------------------------------

st.title("🧠 DocuMind")
st.caption("Intelligent Document Assistant with Multimodal Visual Understanding & Precision Citations")

# Tab Navigation
tab1, tab2 = st.tabs(["📄 Document Assistant", "ℹ️ About & Architecture"])

# ===========================================================================
# TAB 2: ABOUT & ARCHITECTURE
# ===========================================================================
with tab2:
    render_about_tab()

# ===========================================================================
# TAB 1: DOCUMENT ASSISTANT
# ===========================================================================
with tab1:
    # -----------------------------------------------------------------------
    # A. DIRECT WORD CHECK (ON TOP OF THE TAB)
    # -----------------------------------------------------------------------
    st.subheader("🔎 Direct Word & Keyword Check")
    st.caption("Instantly scan all indexed chunks for exact words, phrases, or numerical terms.")

    kw_col1, kw_col2 = st.columns([4, 1])
    with kw_col1:
        keyword_input = st.text_input(
            "Word check directly:",
            placeholder="Type exact keyword, acronym, or number (e.g. 'Transformer', 'QPS', 'Table 4')...",
            label_visibility="collapsed"
        )
    with kw_col2:
        search_kw_btn = st.button("🔍 Exact Search", use_container_width=True)

    if (search_kw_btn or keyword_input) and keyword_input.strip():
        if st.session_state.doc_index is None:
            st.warning("Please upload and process a PDF document first before searching keywords.")
        else:
            matches = rag_pipeline.search_exact_word(keyword_input.strip(), st.session_state.doc_index)
            st.session_state.keyword_results = matches
            
            if matches:
                st.success(f"Found {len(matches)} chunk(s) matching exact keyword: **'{keyword_input.strip()}'**")
                for m_idx, m in enumerate(matches, start=1):
                    with st.expander(
                        f"Match {m_idx} | Chunk: {m['chunk_id']} | Pages: {m['page_start']} - {m['page_end']} | Section: {m['section'] or 'General'}"
                    ):
                        st.write(f"**Snippet Context:** {m['snippet']}")
                        st.markdown("**Full Chunk Content:**")
                        st.code(m['full_content'], language="markdown")
            else:
                st.warning(f"No chunks contained the exact word or phrase: '{keyword_input.strip()}'.")

    st.divider()

    # -----------------------------------------------------------------------
    # B. DOCUMENT UPLOAD & INGESTION
    # -----------------------------------------------------------------------
    if st.session_state.doc_index is None:
        st.subheader("📤 Upload Document")
        
        uploaded_pdf = st.file_uploader(
            "Upload a PDF document to begin intelligent analysis",
            type=["pdf"],
            help="Upload financial reports, technical papers, manuals, or contracts."
        )

        col_proc1, col_proc2 = st.columns([1, 4])
        with col_proc1:
            process_btn = st.button(
                "🚀 Process Document",
                type="primary",
                disabled=uploaded_pdf is None or not (gemini_key and jina_key),
                use_container_width=True
            )
        
        if not (gemini_key and jina_key):
            st.warning("⚠️ Please provide both Gemini and Jina API keys in the sidebar to enable processing.")

        if process_btn and uploaded_pdf is not None:
            # Save uploaded PDF to temporary directory
            temp_dir = Path(tempfile.mkdtemp())
            temp_path = temp_dir / uploaded_pdf.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())

            st.session_state.temp_pdf_path = str(temp_path)
            st.session_state.uploaded_file_name = uploaded_pdf.name

            # Live progress bar and status updates
            progress_bar = st.progress(0, text="Initializing DocuMind ingestion...")
            
            with st.status("Processing document through 6-stage RAG pipeline...", expanded=True) as status_box:
                def on_progress(pct: int, msg: str):
                    progress_bar.progress(pct, text=msg)
                    status_box.write(f"👉 {msg}")

                try:
                    doc_index = rag_pipeline.process_document(
                        pdf_path=str(temp_path),
                        gemini_api_key=gemini_key,
                        jina_api_key=jina_key,
                        progress_callback=on_progress
                    )
                    st.session_state.doc_index = doc_index
                    status_box.update(label="Document successfully processed and indexed!", state="complete", expanded=False)
                    progress_bar.progress(100, text="Done!")
                    st.toast("Document indexed successfully!", icon="✅")
                    st.rerun()
                except Exception as e:
                    status_box.update(label=f"Processing failed: {str(e)}", state="error")
                    st.error(f"Error during document processing: {str(e)}")

    # -----------------------------------------------------------------------
    # C. DOCUMENT DASHBOARD & METRICS
    # -----------------------------------------------------------------------
    if st.session_state.doc_index is not None:
        idx_data = st.session_state.doc_index
        stats = idx_data.stats

        st.subheader("📊 Document Intelligence Metrics")
        
        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
        with m_col1:
            st.metric("Total Pages", stats.total_pages)
        with m_col2:
            st.metric("Total Chunks", stats.total_chunks)
        with m_col3:
            st.metric("Tables Detected", stats.total_tables)
        with m_col4:
            st.metric("Visuals Analyzed", stats.total_pictures)
        with m_col5:
            st.metric("Canonical Elements", stats.total_elements)
        with m_col6:
            st.metric("Index Time", f"{stats.processing_time_sec}s")

        # Document Action Bar
        doc_act_col1, doc_act_col2 = st.columns([4, 1])
        with doc_act_col1:
            st.caption(f"Currently active: **{st.session_state.uploaded_file_name}**")
        with doc_act_col2:
            if st.button("🔄 Process Another Document", use_container_width=True):
                st.session_state.doc_index = None
                st.session_state.messages = []
                st.session_state.uploaded_file_name = None
                st.session_state.temp_pdf_path = None
                st.session_state.keyword_results = None
                st.session_state.current_view_page = 1
                st.rerun()

        st.divider()

        # -------------------------------------------------------------------
        # D. TWO-COLUMN WORKSPACE: CHATBOT + SCROLLABLE PDF VIEWER
        # -------------------------------------------------------------------
        chat_col, pdf_col = st.columns([1, 1], gap="medium")

        # --- COLUMN 1: CONVERSATIONAL CHATBOT ---
        with chat_col:
            st.subheader("💬 Chat with Document")
            st.caption("Ask questions about text paragraphs, financial tables, or visual figures.")

            # Render message history
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("citations"):
                        cites = msg["citations"]
                        badge_parts = []
                        if cites.get("pages"):
                            badge_parts.append(f"📄 Pages: {', '.join(map(str, cites['pages']))}")
                        if cites.get("items"):
                            badge_parts.append(f"📊 Items: {', '.join(cites['items'])}")
                        if cites.get("model"):
                            badge_parts.append(f"🤖 Model: {cites['model']}")
                        if badge_parts:
                            st.caption(" | ".join(badge_parts))

                    # Markdown chunk inspector toggle
                    if msg.get("retrieved_chunks"):
                        with st.expander("🔍 Inspect All Retrieved Chunks (Markdown)"):
                            for r in msg["retrieved_chunks"]:
                                ch = r["chunk"]
                                st.markdown(f"**Source {r['rank']} (Score: {r['score']:.4f}) — Page {ch.get('page_start')}-{ch.get('page_end')}**")
                                st.code(ch["content"], language="markdown")

            # Chat Input
            user_query = st.chat_input("Ask any question from this document...")
            if user_query:
                # Add user question to history
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

                # Generate Answer
                with st.chat_message("assistant"):
                    with st.spinner("Searching FAISS index and generating grounded answer with Gemini..."):
                        try:
                            # 1. Retrieve top_k chunks
                            retrieved = rag_pipeline.retrieve_chunks(
                                query=user_query,
                                index_obj=idx_data,
                                jina_api_key=jina_key,
                                top_k=top_k
                            )
                            
                            # 2. Generate grounded answer
                            answer, model_used, cited_pages, cited_items = rag_pipeline.generate_answer(
                                question=user_query,
                                retrieved_results=retrieved,
                                gemini_api_key=gemini_key,
                                chat_history=st.session_state.messages
                            )

                            # Display answer
                            st.markdown(answer)

                            # Display citations
                            citations_data = {
                                "pages": cited_pages,
                                "items": cited_items,
                                "model": model_used
                            }
                            badge_parts = []
                            if cited_pages:
                                badge_parts.append(f"📄 Pages: {', '.join(map(str, cited_pages))}")
                            if cited_items:
                                badge_parts.append(f"📊 Items: {', '.join(cited_items)}")
                            badge_parts.append(f"🤖 Model: {model_used}")
                            st.caption(" | ".join(badge_parts))

                            # Automatically set PDF viewer to first cited page
                            if cited_pages:
                                st.session_state.current_view_page = cited_pages[0]

                            # Inspect All Chunk Content Markdown Expander
                            with st.expander("🔍 Inspect All Retrieved Chunks (Markdown)"):
                                for r in retrieved:
                                    ch = r["chunk"]
                                    st.markdown(f"**Source {r['rank']} (Similarity: {r['score']:.4f}) — Page {ch.get('page_start')}-{ch.get('page_end')} | Section: {ch.get('section') or 'General'}**")
                                    st.code(ch["content"], language="markdown")

                            # Save to session history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "citations": citations_data,
                                "retrieved_chunks": retrieved
                            })

                        except Exception as exc:
                            err_msg = f"An error occurred while answering: {str(exc)}"
                            st.error(err_msg)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": err_msg
                            })

        # --- COLUMN 2: SCROLLABLE PDF VIEWER ---
        with pdf_col:
            st.subheader("📑 Document Viewer")
            st.caption("Cross-reference answers and inspect original charts & tables.")

            if st.session_state.temp_pdf_path and Path(st.session_state.temp_pdf_path).exists():
                try:
                    pdf_doc = pdfium.PdfDocument(st.session_state.temp_pdf_path)
                    num_pages = len(pdf_doc)

                    # Page navigation controls
                    nav_c1, nav_c2, nav_c3 = st.columns([1, 2, 1])
                    with nav_c1:
                        if st.button("◀ Prev", disabled=st.session_state.current_view_page <= 1, use_container_width=True):
                            st.session_state.current_view_page = max(1, st.session_state.current_view_page - 1)
                            st.rerun()

                    with nav_c2:
                        selected_page = st.number_input(
                            "Page Selector",
                            min_value=1,
                            max_value=max(1, num_pages),
                            value=min(max(1, st.session_state.current_view_page), max(1, num_pages)),
                            step=1,
                            label_visibility="collapsed"
                        )
                        if selected_page != st.session_state.current_view_page:
                            st.session_state.current_view_page = selected_page
                            st.rerun()

                    with nav_c3:
                        if st.button("Next ▶", disabled=st.session_state.current_view_page >= num_pages, use_container_width=True):
                            st.session_state.current_view_page = min(num_pages, st.session_state.current_view_page + 1)
                            st.rerun()

                    st.info(f"Displaying Page **{st.session_state.current_view_page}** of **{num_pages}**")

                    # High-resolution page rendering
                    cur_pg = pdf_doc[st.session_state.current_view_page - 1]
                    pil_image = cur_pg.render(scale=2.0).to_pil()
                    st.image(
                        pil_image,
                        caption=f"Page {st.session_state.current_view_page} — {st.session_state.uploaded_file_name}",
                        use_container_width=True
                    )

                except Exception as pdf_err:
                    st.error(f"Could not render PDF preview: {pdf_err}")
            else:
                st.warning("Original PDF file is not available for preview.")
