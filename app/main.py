import streamlit as st
from ingestion.loader import load_document
from rag.chunker import chunk_text
from rag.embedder import EmbeddingModel
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from llm.llm_service import LLMService
from utils.prompt_builder import build_rag_prompt
from engines.structurer import build_structuring_prompt, parse_structured_output
from engines.risk_engine import assess_risk
from engines.diagnosis_engine import (
    build_diagnosis_prompt,
    parse_diagnosis_output
)

st.set_page_config(page_title="Medical RAG Assistant", layout="wide")
st.title("🏥 Medical RAG Assistant")

# ===============================
# Initialize session state
# ===============================
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "embedder" not in st.session_state:
    st.session_state.embedder = EmbeddingModel()

if "llm" not in st.session_state:
    st.session_state.llm = LLMService()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ===============================
# Sidebar Upload
# ===============================
st.sidebar.header("Upload Medical Document")
uploaded_file = st.sidebar.file_uploader(
    "Supported: PDF, DOCX, CSV, TXT",
    type=["pdf", "docx", "csv", "txt"]
)

if uploaded_file:

    if uploaded_file.size == 0:
        st.sidebar.error("Uploaded file is empty.")
        st.stop()

    try:
        with st.spinner("Processing document..."):

            raw_text = load_document(uploaded_file)

            if not raw_text or len(raw_text.strip()) < 20:
                st.sidebar.error("File unreadable or insufficient text.")
                st.stop()

            chunks = chunk_text(raw_text)
            embeddings = st.session_state.embedder.encode(tuple(chunks))
            dim = len(embeddings[0])

            if st.session_state.vector_store is None:
                st.session_state.vector_store = VectorStore(dim)

            st.session_state.vector_store.add(embeddings, chunks)

        st.sidebar.success("Document indexed successfully!")

    except Exception as e:
        st.sidebar.error("Failed to process document.")
        st.sidebar.text(str(e))

# ===============================
# Chat Interface
# ===============================
st.subheader("Medical Assistant Chat")
user_input = st.chat_input("Ask a medical question...")

if user_input:

    st.session_state.chat_history.append(("user", user_input))
    st.chat_message("user").write(user_input)

    # ==================================================
    # STRUCTURED EXTRACTION MODE
    # ==================================================
    if "extract structured" in user_input.lower():

        if not st.session_state.vector_store:
            st.warning("Please upload a medical document for structured extraction.")
            st.stop()

        retriever = Retriever(
            st.session_state.embedder,
            st.session_state.vector_store
        )

        retrieved_chunks = retriever.retrieve(user_input)
        prompt = build_structuring_prompt(retrieved_chunks)

        try:
            with st.spinner("Extracting structured medical data..."):
                raw_response = st.session_state.llm.generate(prompt)
        except Exception:
            st.error("LLM failed to generate response.")
            st.stop()

        structured_report = parse_structured_output(raw_response)

        if structured_report:

            risk_result = assess_risk(structured_report)

            st.subheader("📊 Structured Medical Data")
            st.json(structured_report.dict())

            st.subheader("⚠ Risk Assessment")

            if risk_result["risk_level"] == "HIGH":
                st.error("HIGH RISK — Immediate medical consultation advised.")
            elif risk_result["risk_level"] == "MODERATE":
                st.warning("MODERATE RISK — Monitor symptoms carefully.")
            else:
                st.success("LOW RISK — No immediate danger detected.")

            st.write("### Reasons:")
            for reason in risk_result["reasons"]:
                st.write(f"- {reason}")

        else:
            st.error("Failed to parse structured data.")

        st.stop()

    # ==================================================
    # DIAGNOSIS MODE (Works WITH or WITHOUT document)
    # ==================================================
    if "diagnose" in user_input.lower():

        if st.session_state.vector_store:
            retriever = Retriever(
                st.session_state.embedder,
                st.session_state.vector_store
            )
            retrieved_chunks = retriever.retrieve(user_input)
            context = "\n".join(retrieved_chunks)
        else:
            context = "No medical document provided. Use general medical reasoning."

        prompt = build_diagnosis_prompt(context, user_input)

        try:
            with st.spinner("Generating differential diagnosis..."):
                raw_response = st.session_state.llm.generate(prompt)
        except Exception:
            st.error("LLM failed to generate diagnosis.")
            st.stop()

        diagnosis = parse_diagnosis_output(raw_response)

        if diagnosis:
            st.subheader("🩺 Primary Suspected Condition")
            st.info(diagnosis.primary_suspected_condition)

            st.subheader("📋 Differential Diagnosis")
            for item in diagnosis.differential_diagnosis:
                st.write(f"### {item.condition}")
                st.write(f"Likelihood: {item.likelihood}")
                st.write(f"Reason: {item.reason}")
                st.write("---")

            st.subheader("🧪 Recommended Tests")
            for test in diagnosis.recommended_tests:
                st.write(f"- {test}")

        else:
            st.error("Failed to generate structured diagnosis.")

        st.stop()

    # ==================================================
    # NORMAL CHAT MODE (RAG or General Medical)
    # ==================================================
    if st.session_state.vector_store:

        retriever = Retriever(
            st.session_state.embedder,
            st.session_state.vector_store
        )

        retrieved_chunks = retriever.retrieve(user_input)

        # Inject conversation memory
        history_context = ""
        for role, msg in st.session_state.chat_history[-5:]:
            history_context += f"{role.upper()}: {msg}\n"

        prompt = build_rag_prompt(
            retrieved_chunks,
            user_input + "\n\nConversation History:\n" + history_context
        )

    else:
        prompt = f"""
You are a medical AI assistant.

The user has NOT uploaded any medical document.

Answer using general medical knowledge.

User Question:
{user_input}

Rules:
- Provide educational information only.
- Do not give final diagnosis.
- If symptoms seem severe, advise emergency care.
- Add disclaimer recommending consultation with a doctor.
"""

    try:
        with st.spinner("Generating response..."):
            response = st.session_state.llm.generate(prompt)
    except Exception:
        st.error("LLM failed to generate response.")
        st.stop()

    st.session_state.chat_history.append(("assistant", response))
    st.chat_message("assistant").write(response)