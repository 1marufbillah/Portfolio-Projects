import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import tempfile
import os

# Custom CSS for improved styling
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stTextInput>div>div>input {border: 2px solid #4CAF50;}
    .stFileUploader>div>div>div>div>div>div {border: 2px dashed #4CAF50;}
    .response-box {padding: 20px; background-color: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 20px;}
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 DeepSeek-R1 RAG Assistant")
st.markdown("### AI-powered Document Analysis with Retrieval-Augmented Generation")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Choose LLM Model", ["deepseek-r1:7b", "llama3.2", "mistral"])
    temperature = st.slider("Creativity Level", 0.0, 1.0, 0.7)
    top_k = st.slider("Number of Retrieved Chunks", 1, 5, 3)
    st.markdown("---")
    st.markdown("**Note:** Ensure Ollama is running locally with selected model")

# Initialize components once
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def initialize_llm(model_name, temperature):
    return Ollama(model=model_name, temperature=temperature)

embedder = load_embeddings()

# File upload section
with st.container():
    uploaded_file = st.file_uploader("📤 Upload PDF Document", type="pdf")

# Processing pipeline
if uploaded_file and ("vector_store" not in st.session_state or st.sidebar.button("Reload Document")):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    with st.status("🧠 Processing Document...", expanded=True) as status:
        try:
            st.write("📄 Loading PDF...")
            loader = PDFPlumberLoader(tmp_file_path)
            docs = loader.load()

            st.write("🔪 Splitting document...")
            text_splitter = SemanticChunker(embedder)
            documents = text_splitter.split_documents(docs)

            st.write("🔮 Generating embeddings...")
            vector_store = FAISS.from_documents(documents, embedder)
            st.session_state.retriever = vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": top_k}
            )

            status.update(label="✅ Document Processing Complete!", state="complete")
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
        finally:
            os.unlink(tmp_file_path)

# Q&A Section
if "retriever" in st.session_state:
    llm = initialize_llm(model_name, temperature)
    
    # Enhanced prompt template
    prompt_template = """
    ### Instruction:
    You are an expert document analyst. Use the following context to answer the question.
    Follow these rules:
    1. Answer directly and concisely (3-4 sentences max)
    2. Use simple, clear language
    3. If uncertain, say "I'm not sure based on the document"
    4. Always reference page numbers when possible

    ### Context:
    {context}

    ### Question:
    {question}

    ### Response:
    """

    QA_PROMPT = PromptTemplate.from_template(prompt_template)

    # Configure QA system
    qa_system = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )

    # Chat interface
    with st.form("qa_form"):
        question = st.text_input("💬 Ask about the document:", placeholder="Type your question here...")
        submitted = st.form_submit_button("🚀 Get Answer")

    if submitted and question:
        with st.spinner("🤖 Analyzing document..."):
            try:
                result = qa_system.invoke({"query": question})
                
                with st.container():
                    st.markdown("---")
                    st.markdown(f"### 📝 Answer:")
                    st.info(result["result"])
                    
                    with st.expander("🔍 View Source References"):
                        for i, doc in enumerate(result["source_documents"], 1):
                            st.markdown(f"""
                            **Source {i}** (Page {doc.metadata.get('page', 'N/A')}):
                            ```
                            {doc.page_content[:300]}...
                            ```
                            """)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

elif uploaded_file is None:
    st.info("👋 Please upload a PDF document to get started")
