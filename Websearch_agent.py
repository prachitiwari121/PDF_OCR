import requests
import streamlit as st
from transformers import pipeline

#----------------------------------------------------- SEARCH AND QA MODEL INITIALIZATION --------------------------------
search_pipeline = pipeline("search")
qa_pipeline = pipeline("question-answering")

#----------------------------------------------------- WEB SEARCH FUNCTION ----------------------------------------------
def perform_web_search(query):
    """Perform a web search using a search API."""
    search_url = "https://api.example.com/search"
    response = requests.get(search_url, params={"q": query})
    
    if response.ok:
        return response.json().get("results", [])
    else:
        return []

#----------------------------------------------------- QA EXTRACTION FUNCTION -------------------------------------------
def extract_answer_from_context(context, question):
    """Use a QA pipeline to extract an answer from context."""
    return qa_pipeline(question=question, context=context)

#----------------------------------------------------- STREAMLIT APP CONFIGURATION --------------------------------------
st.set_page_config(page_title="Web Search and QA", page_icon="🔍")
st.title("Web Search and Question Answering 🔍")

#----------------------------------------------------- USER INPUT FOR SEARCH --------------------------------------------
search_query = st.text_input("Enter your search query:")

if st.button("Search"):
    if search_query:
        search_results = perform_web_search(search_query)
        
        if search_results:
            st.success("Search completed successfully!")
            for result in search_results:
                st.write(f"**Title:** {result['title']}")
                st.write(f"**Link:** [Visit]({result['link']})")
                st.write(f"**Snippet:** {result['snippet']}\n")
        else:
            st.error("No results found.")
    else:
        st.warning("Please enter a search query.")

#----------------------------------------------------- QUESTION ANSWERING BASED ON SEARCH --------------------------------
user_question = st.text_input("Ask a question about the search results:")

if user_question and search_results:
    context = " ".join([result['snippet'] for result in search_results])
    
    answer = extract_answer_from_context(context, user_question)
    
    st.write("### Answer:")
    st.write(answer.get("answer", "I'm sorry, I couldn't find an answer to that."))