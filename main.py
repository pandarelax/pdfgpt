import streamlit as st

def main():
    st.set_page_config(page_title="PdfGPT App", layout="wide")
    
    st.header("Chat With Your PDFs :books:")
    st.text_input("Ask a question about your PDFs")
    
if __name__ == "__main__":
    main()