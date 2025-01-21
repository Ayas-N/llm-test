from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
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

retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs={"k":5})
llm = ChatGoogleGenerativeAI(model = "gemini-1.5-pro", temperature = 0.3, max_tokens = 5000)
system_prompt = """You are an assistant for question-answering tasks.
Answer these questions as if you are talking to a Bioinformatics expert. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer say you don't know.
Context: {context} 
Answer:"""

prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", "{input}")])
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

response = rag_chain.invoke({"input":"What is Task Decomposition"})
loader = PyPDFLoader("pdfs/banksy.pdf")

# Consider adding in supplementary information to help answer the questions.
# Does looking at the website help?

def retrieve_document(loader, chunk_size=500):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size)
    chunks = text_splitter.split_documents(loader.load())
    db = Chroma.from_documents(chunks, embeddings, persist_directory='./chroma_db_')
    db_connection = Chroma(persist_directory="./chroma_db_", embedding_function = embeddings)
    retriever = db_connection.as_retriever(search_type = "similarity", search_kwargs = {"k":10})
    return retriever

def generate_document(loader, question):
    retriever = retrieve_document(loader)
    retrieved_docs = retriever.invoke(question)
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    response = rag_chain.invoke({"input":question})
    print(response["answer"])

retrieve_document(loader)
generate_document(loader, """I am a bioinformatician who wishes to learn more about the spatial clustering methods employed in scientific literature. From here on out, using only information from attached article as context, talk to me as if I am an expert and answer the following question:

What are some method assumptions employed by BANKSY?
""")
