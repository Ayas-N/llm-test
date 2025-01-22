import streamlit as st
import time
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import create_retrieval_chain
import bs4
import os
from io import StringIO

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = open(file= 'langsmith_apikey.txt').read()
api_key = open('gemini_apikey.txt').read()
llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash", temperature = 0.3, max_tokens = 5000, api_key = api_key)
system_prompt = """You are an assistant for question-answering tasks.
Answer these questions as if you are talking to a Bioinformatics expert that specialises in spatial transcriptomics. 
Use the following pieces of retrieved context to answer the question. 
Feel free to use external sources of information to help reach your conclusion.
If you don't know the answer say you don't know. Let's think step by step.
Make sure your answer is technical, but concise.

Provide
Context: {context} 
Answer:"""

embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key = api_key)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", "{input}")])

with st.chat_message("user"):
    st.write("Hello")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def generate_response(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def retrieve_document(filename, chunk_size=3200):
    docs = os.listdir("tmp")
    for doc in docs:
        print(doc)
    loader = PyPDFLoader(f"tmp/{filename}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size)
    chunks = text_splitter.split_documents(loader.load())
    Chroma.from_documents(chunks, embeddings, persist_directory='./chroma_db_')
    db_connection = Chroma(persist_directory="./chroma_db_", embedding_function = embeddings)
    retriever = db_connection.as_retriever(search_type = "similarity", search_kwargs = {"k":5})
    bm25_retriever = BM25Retriever.from_texts([t.page_content for t in chunks])
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever],
        weights=[0.5, 0.5])

    return ensemble_retriever

def generate_document(filename, question):
    retriever = retrieve_document(filename)
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    with st.chat_message("assistant"):
        response = st.write_stream(generate_response(f"Uploaded file name: {uploaded_file.name}"))
        st.session_state.messages.append({"role":"assistant", "content":response})
    response = rag_chain.invoke({"input":question})
    return response["answer"]

uploaded_file = st.file_uploader("Upload an article", type=('pdf'))
if uploaded_file is not None:
    filename = uploaded_file.name
    # Can I write the files to a temp storage? Answer: Yes!
    with open(f'tmp/{filename}', 'wb') as f:
        print("File has been written")
        f.write(uploaded_file.getvalue())
    

# Respond to user
if usr_input := st.chat_input():
    # Display user message in chat message container
    st.chat_message("user").markdown(usr_input)
    llm_answer = generate_document(filename, usr_input)
    response = st.write_stream(generate_response(llm_answer))