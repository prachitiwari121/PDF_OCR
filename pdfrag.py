from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PIL import Image
import pytesseract
import streamlit as st



#----------------------------------------------------- TEXT EXTRACTION----------------------------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

#----------------------------------------------------- TEXT SPLITTING ----------------------------------------------------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

#----------------------------------------------------- FAISS VECTOR STORE CREATION ---------------------------------------
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    
    if not text_chunks:
        raise ValueError("No text chunks provided for vector store creation.")
    
    print(f"Number of text chunks: {len(text_chunks)}")
    

    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        print(f"Error creating vector store: {e}")
        raise

#----------------------------------------------------- GOOGLE GEMINI CONVERSATIONAL CHAIN --------------------------------
def get_conversational_chain():
    prompt_template = """
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

#----------------------------------------------------- USER INPUT HANDLER ------------------------------------------------
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    
    try:
        # new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        new_db = FAISS.load_local("faiss_index", embeddings)

        docs = new_db.similarity_search(user_question)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply: ", response["output_text"])

#----------------------------------------------------- STREAMLIT APP LOGIC -----------------------------------------------
def main():
    st.set_page_config("Chat PDF", layout="wide")
    st.header("Chat with PDF using Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        
        if pdf_docs and st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                
                if raw_text.strip():
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDF processing completed successfully!")
                else:
                    st.warning("No text found in the uploaded PDFs. Please check the files and try again.")

#----------------------------------------------------- MAIN ENTRY POINT --------------------------------------------------
if __name__ == "__main__":
    main()




