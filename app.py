__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import create_retrieval_chain
from langchain_community.tools.tavily_search import TavilySearchResults
from prompts import prompt
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = open(file= 'langsmith_apikey.txt').read()
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = open('tavilyapi.txt').read()
api_key = open('gemini_apikey.txt').read()
llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash", temperature = 0.3, max_tokens = 5000, api_key = api_key)
embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key = api_key)
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def retrieve_document(filename, chunk_size=3200):
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

def generate_document(filename):
    '''Creates a tool that extracts relevant information from pdf files existing in the tmp folder.'''
    print(filename)
    retriever = retrieve_document(filename)
    as_tool = retriever.as_tool(
        name= "pdf_read",
        description = "Use this tool to get relevant information from the delivered pdf to answer the input question."
    )

    return as_tool

uploaded_file = st.file_uploader("Upload an article", type=('pdf'), accept_multiple_files= True)

for file in uploaded_file:
    filename = file.name
    # Can I write the files to a temp storage? Answer: Yes!
    with open(f'tmp/{filename}', 'wb') as f:
        print("File has been written")
        f.write(file.getvalue())

agent_prompt = PromptTemplate(template= prompt)
memory = ChatMessageHistory(session_id="test-session")

def respond(usr_input):
    if 'agent_history' not in st.session_state:
        tools = [generate_document(pdf.name) for pdf in uploaded_file]
        tools.append(TavilySearchResults(max_results = 5))
        agent = create_react_agent(llm, tools, agent_prompt)
        # Create an agent executor by passing in the agent and tools
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        st.session_state['agent_history'] = RunnableWithMessageHistory(
            agent_executor,
            lambda session_id: memory,
            input_messages_key= "input",
            history_messages_key= "chat_history",
            )
    response = st.session_state['agent_history'].invoke({"input":usr_input,},
                        config={"configurable": {"session_id": "<foo>"}})
    return response

# Respond to user
if usr_input := st.chat_input():
    st.chat_message("user").markdown(usr_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": usr_input})   
    with st.spinner('Processing...'):
        llm_answer = respond(usr_input)["output"]
        response = st.chat_message("assistant").markdown(llm_answer)
    # Add user message to chat history
    st.session_state.messages.append({"role": "assistant", "content": llm_answer})   
