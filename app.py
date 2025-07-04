import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

import os

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()
print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=1000,
     )
    chunks=text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore=FAISS.from_texts(text_chunks,embedding=embeddings)
    vectorstore.save_local("faiss_index")#folder to save the vectorstore
    

def get_conversation_chain():
    prompt_template = """
    You are a helpful and knowledgeable assistant. Use only the information in the context below to answer the user's question clearly and concisely.

    - If the context does **not** contain the answer, reply with: "The answer is not available in the provided context."
    - Do **not** make up any information.
    - Keep your response factual and relevant.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model=ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.3)
    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain=get_conversation_chain()
    response=chain({"input_documents":docs,"question":user_question}
                   ,return_only_outputs=True)
    print(response)

    st.write(response["output_text"])

def main():
    st.set_page_config(page_title="Gemini PDF Chatbot",page_icon=":books:")
    st.header("Gemini PDF Chatbot")
    user_question=st.text_input("Ask a question about the PDF:")

    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("Your documents")

        pdf_docs=st.file_uploader("Upload your PDFs here and click on 'Process'",type="pdf",accept_multiple_files=True)
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF before processing.")
                return
            with st.spinner("Processing"):
                raw_text=get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.warning("No text could be extracted from the uploaded PDF(s).")
                    return
                text_chunks=get_text_chunks(raw_text)
                if not text_chunks:
                    st.warning("No text chunks could be created from the extracted text.")
                    return
                get_vectorstore(text_chunks)
                st.success("Done!")

if __name__=="__main__":
    main()