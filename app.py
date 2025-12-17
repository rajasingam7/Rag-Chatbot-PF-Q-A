import streamlit as st
import time

# ================================
# LANGCHAIN IMPORTS (CLASSIC)
# ================================

from langchain_groq import ChatGroq
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings

# ================================
# STREAMLIT UI
# ================================

st.set_page_config(page_title="RAG PDF Q&A", layout="wide")
st.title("📄 RAG Document Q&A with Groq + Llama3")

# ================================
# GROQ API KEY (SAFE)
# ================================

groq_api_key = st.text_input(
    "Enter your Groq API Key",
    type="password"
)

if not groq_api_key:
    st.warning("Please enter your Groq API Key to continue.")
    st.stop()

# ================================
# LLM
# ================================

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)



# ================================
# EMBEDDINGS
# ================================

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ================================
# PROMPT
# ================================

prompt = ChatPromptTemplate.from_template(
    """
    You are an AI assistant.
    Answer ONLY using the provided context.
    If the answer is not in the context, say "I don't know".

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# ================================
# VECTOR CREATION (SAFE)
# ================================

def create_vector_embedding():
    loader = PyPDFDirectoryLoader("research_papers")
    docs = loader.load()

    if not docs:
        st.error("❌ No PDFs found in 'research_papers' folder.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    final_docs = splitter.split_documents(docs)

    if not final_docs:
        st.error("❌ PDFs loaded but no text could be extracted (scanned PDFs?).")
        return

    vectors = FAISS.from_documents(
        final_docs,
        embeddings
    )

    st.session_state.vectors = vectors
    st.success(f"✅ Vector DB created with {len(final_docs)} chunks")

# ================================
# USER INPUT
# ================================

user_prompt = st.text_input("Ask a question from the research papers")

if st.button("📥 Create Document Embeddings"):
    create_vector_embedding()

# ================================
# RAG EXECUTION
# ================================

if user_prompt:

    if "vectors" not in st.session_state:
        st.warning("Please create document embeddings first.")
        st.stop()

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    elapsed = time.process_time() - start

    st.subheader("✅ Answer")
    st.write(response["answer"])
    st.caption(f"⏱ Response time: {elapsed:.2f} seconds")

    with st.expander("📚 Retrieved Document Chunks"):
        for i, doc in enumerate(response["context"], 1):
            st.markdown(f"**Chunk {i}**")
            st.write(doc.page_content)
            st.write("-----")
