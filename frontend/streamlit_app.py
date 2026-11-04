import streamlit as st
import tempfile
import os
import hashlib

from backend.app.rag.pipeline import build_vector_store
from backend.app.rag.pipeline import answer_question

st.set_page_config(
    page_title="DocuMind",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
div.stFormSubmitButton > button {
    background-color: #A78BFA;
    color: #171421;
    border: 1px solid #A78BFA;
    border-radius: 8px;
    transition: all 0.2s ease;
}

div.stFormSubmitButton > button:hover {
    background-color: #8B6DE0;
    border-color: #8B6DE0;
    color: #FFFFFF;
}

/* Reduce top whitespace */
.block-container {
    padding-top: 2rem !important;
}
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("📄 DocuMind")
st.markdown(
    "##### Your documents. Your questions. AI-powered answers."
)
st.caption(
    "An intelligent PDF assistant powered by Retrieval-Augmented Generation."
)

st.divider()

st.header("Your document")
st.caption("Upload a PDF to start a conversation with your document.")

with st.container(border=True):

    uploaded_file = st.file_uploader(
        "Upload or replace your PDF",
        type=["pdf"],
        help="Choose a PDF document to analyze."
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        # Process only if this is a new document
        if (
            st.session_state.get("processed_file_hash") != file_hash
            or "vector_store" not in st.session_state
        ):

            # Clear previous document data
            st.session_state.pop("vector_store", None)
            st.session_state.chat_history = []

            temp_file_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(file_bytes)
                    temp_file_path = temp_file.name

                with st.spinner("Processing your document..."):
                    vector_store = build_vector_store(temp_file_path)

                st.session_state.vector_store = vector_store
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.processed_file_hash = file_hash

            except Exception as e:
                st.error(f"Error processing PDF: {e}")

            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        # Show document status
        if (
            st.session_state.get("processed_file_hash") == file_hash
            and "vector_store" in st.session_state
        ):
            st.success("Document ready")
            st.write(f"📄 **{uploaded_file.name}**")

st.header("Ask your document")
st.caption(
    "Ask questions, explore concepts, and get answers grounded in your PDF."
)

if "vector_store" in st.session_state:

    # Question input form
    with st.container(border=True):

        st.markdown("**What would you like to know?**")

        with st.form("question_form", clear_on_submit=True):

            question = st.text_input(
                "Your question",
                placeholder="e.g. Explain the main concepts in this document.",
                label_visibility="collapsed"
            )

            submitted = st.form_submit_button(
                "Ask DocuMind",
                use_container_width=True
            )

    # Process the submitted question
    if submitted:

        if question.strip():

            try:
                with st.spinner("Finding the answer..."):

                    result = answer_question(
                        st.session_state.vector_store,
                        question
                    )

                # Save this conversation
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"]
                })

            except RuntimeError as e:
                st.error(str(e))

            except Exception:
                st.error(
                    "Something went wrong while generating your answer. "
                    "Please try again later."
                )

        else:
            st.warning("Please enter a question.")

    # Display conversation history
    if st.session_state.chat_history:

        st.divider()

        col1, col2 = st.columns([5, 1])

        with col1:
            st.subheader("Conversation")

        with col2:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        for chat in reversed(st.session_state.chat_history):

            # Display user's question
            with st.chat_message("user"):
                st.write(chat["question"])

            # Display assistant's answer
            with st.chat_message("assistant"):

                st.write(chat["answer"])

                st.markdown("**Sources**")

                for source in chat["sources"]:

                    with st.expander(f"📄 Page {source['page']}"):
                        st.caption(
                            f"Retrieval similarity: {source['similarity']:.4f}"
                        )
                        st.write(source["text"])

else:
    st.info("Upload a PDF to start asking questions.")