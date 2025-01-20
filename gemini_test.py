from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
import bs4

from sys import argv
import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = open(file= 'langsmith_apikey.txt').read()
os.environ["GOOGLE_API_KEY"] = open('gemini_apikey.txt').read()
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000)
docs = text_splitter.split_documents(loader.load())

vector_store = Chroma.from_documents(documents = docs, embedding = embeddings)

retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs={"k":10})

retrieved_docs = retriever.invoke("What is Task Decomposition?")
llm = ChatGoogleGenerativeAI(model = "gemini-1.5-pro", temperature = 0.3, max_tokens = 250)
system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Context: {context} 
Answer:"""

prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", "{input}")])
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

response = rag_chain.invoke({"input":"What is Task Decomposition"}
                      )

print(response["answer"])