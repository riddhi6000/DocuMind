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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("📄 DocuMind")
st.subheader("Your AI-powered PDF assistant")

st.write(
    "Upload a PDF document and ask questions about its contents."
)

st.divider()

st.header("Upload your document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    # Generate a unique fingerprint for the uploaded PDF
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # Process only if this is a new document
    if (
        st.session_state.get("processed_file_hash") != file_hash
        or "vector_store" not in st.session_state
    ):

        # Clear the previous document's data
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

            with st.spinner("Processing your PDF..."):

                vector_store = build_vector_store(temp_file_path)

            # Save the processed document
            st.session_state.vector_store = vector_store
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.processed_file_hash = file_hash

            st.success(
                "Your document is ready! You can now ask questions."
            )

        except Exception as e:
            st.error(f"Error processing PDF: {e}")

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    else:
        st.success("Your document is ready! You can now ask questions.")

st.header("Ask your document")

if "vector_store" in st.session_state:

    # Display a button to clear the conversation
    if st.button("Clear conversation"):
        st.session_state.chat_history = []

    # Question input form
    with st.form("question_form", clear_on_submit=True):

        question = st.text_input(
            "Enter your question",
            placeholder="What does this document explain?"
        )

        submitted = st.form_submit_button("Ask DocuMind")

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

            except Exception as e:
                st.error(f"Error generating answer: {e}")

        else:
            st.warning("Please enter a question.")

    # Display conversation history
    if st.session_state.chat_history:

        st.divider()
        st.subheader("Conversation")

        for chat in st.session_state.chat_history:

            # Display user's question
            with st.chat_message("user"):
                st.write(chat["question"])

            # Display assistant's answer
            with st.chat_message("assistant"):

                st.write(chat["answer"])

                st.markdown("**Sources**")

                for source in chat["sources"]:

                    with st.expander(
                        f"Page {source['page']} | "
                        f"Similarity: {source['similarity']:.4f}"
                    ):
                        st.write(source["text"])

else:
    st.info("Upload a PDF to start asking questions.")