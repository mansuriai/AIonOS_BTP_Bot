# app/upload_app.py
import streamlit as st
from pathlib import Path
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from fix_ssl import *
from utils.config import config
from core.document_processor import EnhancedDocumentProcessor
from core.embeddings import EmbeddingManager
from core.vector_store import VectorStore

# # Initialize session state
# if "uploaded_files" not in st.session_state:
#     st.session_state.uploaded_files = []

# Initialize components
doc_processor = EnhancedDocumentProcessor()
embedding_manager = EmbeddingManager()
vector_store = VectorStore()

st.set_page_config(
    page_title=f"{config.APP_TITLE} - Document Upload",
    layout="wide"
)

st.title(f"{config.APP_TITLE} - Document Upload")

# File upload section
st.header("Upload Documents")
uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=['pdf'],
    accept_multiple_files=True
)

if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

if st.session_state.uploaded_files:
    st.write(f"{len(st.session_state.uploaded_files)} documents ready for processing.")

    # Add better error handling and feedback
    if st.button("Process Documents"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            total_files = len(st.session_state.uploaded_files)
            for i, file in enumerate(st.session_state.uploaded_files):
                status_text.text(f"Processing file {i+1} of {total_files}: {file.name}")
                
                # Save PDF to storage directory
                pdf_path = config.PDF_STORAGE_DIR / file.name
                with open(pdf_path, 'wb') as f:
                    f.write(file.getvalue())
                
                # Process for vector store
                file_path = config.DATA_DIR / file.name
                with open(file_path, 'wb') as f:
                    f.write(file.getvalue())
                
                # Process documents
                chunks = doc_processor.process_file(file_path)
                
                # Generate embeddings
                embeddings = embedding_manager.generate_embeddings(
                    [chunk['text'] for chunk in chunks]
                )
                
                # Store in vector database
                vector_store.add_documents(chunks, embeddings)
                
                progress_bar.progress((i + 1) / total_files)
            
            st.success("All documents processed and indexed successfully!")
            st.session_state.uploaded_files = []
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            st.stop()
    
    # if st.button("Process Documents"):
    #     with st.spinner("Processing documents..."):
    #         for file in st.session_state.uploaded_files:
    #             # Save PDF to storage directory
    #             pdf_path = config.PDF_STORAGE_DIR / file.name
    #             with open(pdf_path, 'wb') as f:
    #                 f.write(file.getvalue())
                
    #             # Process for vector store
    #             file_path = config.DATA_DIR / file.name
    #             with open(file_path, 'wb') as f:
    #                 f.write(file.getvalue())
                
    #             # Process documents
    #             chunks = doc_processor.process_file(file_path)
                
    #             # Generate embeddings
    #             embeddings = embedding_manager.generate_embeddings(
    #                 [chunk['text'] for chunk in chunks]
    #             )
                
    #             # Store in vector database
    #             vector_store.add_documents(chunks, embeddings)
            
    #         st.success("Documents processed and indexed!")
    #         st.session_state.uploaded_files = []
