import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def load_and_process_pdf(pdf_path):
    """Load and process PDF document."""
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20
        )
        documents = text_splitter.split_documents(docs)
        return documents
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def setup_retrieval_chain():
    """Setup the retrieval chain with FAISS and Ollama."""
    try:
        # Initialize embeddings and vector store
        embeddings = OllamaEmbeddings(model="llama3.2")

        # Load documents and create FAISS index
        documents = load_and_process_pdf("attention.pdf")
        if documents is None:
            return None

        db = FAISS.from_documents(documents[:30], embeddings)

        # Setup LLM and prompt
        llm = Ollama(model="llama3.2")
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context.
        Think step by step before providing a detailed answer.
        I will tip you \$1000 if the user finds the answer helpful.
        <context>
        {context}
        </context>
        Question: {input}""")

        # Create chains
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = db.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        return retrieval_chain, db
    except Exception as e:
        st.error(f"Error setting up retrieval chain: {str(e)}")
        return None, None

def main():
    st.set_page_config(page_title="PDF Question Answering", layout="wide")

    st.title("📚 PDF Question Answering System")
    st.markdown("Ask questions about the attention mechanism paper!")

    # Initialize the retrieval chain
    retrieval_chain, db = setup_retrieval_chain()

    if retrieval_chain is None:
        st.error("Failed to initialize the system. Please check if all requirements are met.")
        return

    # User input
    user_question = st.text_input(
        "Ask a question about the paper:",
        placeholder="e.g., Explain Scaled Dot-Product Attention"
    )

    if user_question:
        with st.spinner("Searching and generating answer..."):
            try:
                # Simple similarity search
                results = db.similarity_search(user_question)
                st.subheader("📑 Relevant Context:")
                with st.expander("Show context"):
                    st.markdown(results[0].page_content)

                # Generate detailed answer
                response = retrieval_chain.invoke({"input": user_question})

                st.subheader("🤖 Answer:")
                st.markdown(response['answer'])

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application uses:
        - llama3.2 for text generation
        - FAISS for similarity search
        - LangChain for chain management
        - PDF loader for document processing
        """)

if __name__ == "__main__":
    main()