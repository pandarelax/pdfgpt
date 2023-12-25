from ast import List
from openai import embeddings
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def get_pdf_text(pdfs) -> str:
    text: str = ""
    
    for pdf in pdfs:
        reader: PdfReader = PdfReader(pdf)
        for page in reader.pages:
            text+= page.extractText()
    return text

def get_text_chunks(text) -> List[str]:
    splitter: CharacterTextSplitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks: List[str] = splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings()  # noqa: F811
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store
    

def main() -> None:
    load_dotenv()
    
    st.set_page_config(page_title="PdfGPT App", layout="wide")
    st.header("Chat With Your PDFs :books:")
    st.text_input("Ask a question about your PDFs")
    
    with st.sidebar:
        st.subheader("Your Documents")
        pdfs = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)
        if pdfs:
            with st.spinner("Processing PDFs..."):
                # Get pdf text
                raw_text = get_pdf_text(pdfs)

                # Get the chunks from text
                text_chunks = raw_text.split("\n\n")
                
                # Create vector store
                vector_store = get_vector_store(text_chunks)
        
if __name__ == "__main__":
    main()