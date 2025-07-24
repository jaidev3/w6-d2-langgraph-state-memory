import streamlit as st
from doc_processing import DocumentProcessor

st.title("Banking Knowledge Base")

processor = DocumentProcessor()
tab1, tab2 = st.tabs(["Add & Process Documents", "User Query"])

with tab1:
    st.header("Add and Process Documents")
    uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True, type=["pdf", "txt", "docx"])
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded.")
        # Placeholder for document processing logic
        if st.button("Process Documents"):
            docs = processor.process_pdf_files(uploaded_files)
            st.info(f"Processed {len(docs)} documents")
            chunks = processor.split_documents(docs)
            st.info(f"Split {len(chunks)} chunks")
            processor.save_to_pinecone(chunks)
            st.info("Chunks saved to Pinecone")

with tab2:
    st.header("Ask a Question")
    user_query = st.text_input("Enter your query:")
    if st.button("Submit Query"):
        if user_query.strip():
            # Placeholder for query processing logic
            st.info(f"Processing query: {user_query} (functionality to be implemented)")
        else:
            st.warning("Please enter a query before submitting.")