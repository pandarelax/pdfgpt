import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from template import css, bot_template, user_template
from langchain.llms import HuggingFaceHub


def get_pdf_text(pdfs) -> str:
    text: str = ""
    
    for pdf in pdfs:
        reader: PdfReader = PdfReader(pdf)
        for page in reader.pages:
            text+= page.extractText()
    return text

def get_text_chunks(text) -> list[str]:
    splitter: CharacterTextSplitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks: list[str] = splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # Production embeddings
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings()
    
    # Local embeddings for testing
    # embeddings: HuggingFaceInstructEmbeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/inctructor-xl")  # noqa: F811
    
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def create_conversation_chain(vector_store) -> ConversationalRetrievalChain:
    # OpenAI LLM
    llm: ChatOpenAI = ChatOpenAI()
    
    # HuggingFace LLM
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    
    memory: ConversationBufferMemory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
    conversation_chain: ConversationalRetrievalChain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def handle_user_input(user_question: str):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main() -> None:
    load_dotenv()
    
    st.set_page_config(page_title="PdfGPT App", layout="wide")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat With Your PDFs :books:")
    user_question: str = st.text_input("Ask a question about your PDFs")
    if user_question:
        handle_user_input(user_question)
    
    with st.sidebar:
        st.subheader("Your Documents")
        pdfs = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)
        if pdfs:
            if st.button("Process"):
                with st.spinner("Processing PDFs..."):
                    # Get pdf text
                    raw_text = get_pdf_text(pdfs)

                    # Get the chunks from text
                    text_chunks = raw_text.split("\n\n")
                    
                    # Create vector store
                    vector_store = get_vector_store(text_chunks)
                    
                    # Create conversation chain
                    st.session_state.conversation = create_conversation_chain(vector_store)
        
if __name__ == "__main__":
    main()