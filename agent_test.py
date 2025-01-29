from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import bs4

import os
os.environ["GOOGLE_API_KEY"] = open('gemini_apikey.txt').read()
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = open('tavilyapi.txt').read()

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash", temperature = 0.5, max_tokens = 5000)
embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
system_prompt = """You are an assistant for question-answering tasks.
Answer these questions as if you are talking to a Bioinformatics expert that specialises in spatial transcriptomics. 
Use the following pieces of retrieved context to answer the question. 
Feel free to use external sources of information to help reach your conclusion.
If you don't know the answer say you don't know. Let's think step by step.
Make sure your answer is technical, but concise.

Provide
Context: {context} 
Answer:"""

prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", "{input}")])


def retrieve_document(loader):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 3200)
    chunks = text_splitter.split_documents(loader.load())
    db = Chroma.from_documents(chunks, embeddings, persist_directory='./chroma_db_')
    db_connection = Chroma(persist_directory="./chroma_db_", embedding_function = embeddings)
    retriever = db_connection.as_retriever(search_type = "similarity", search_kwargs = {"k":5})
    bm25_retriever = BM25Retriever.from_texts([t.page_content for t in chunks])
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever],
        weights=[0.5, 0.5])
    return ensemble_retriever

def generate_document(filename):
    loader = PyPDFLoader(f"pdfs/{filename}.pdf")
    retriever = retrieve_document(loader)
    as_tool = retriever.as_tool(
        name= "pdf_read",
        description = "Use this tool to get relevant information from the delivered pdf to answer the input question."
    )

    return as_tool

prompt = """Assistant is a large language model trained by Google.

Assistant is an expert in spatial transcriptomics, from answering simple questions to providing in-depth explanations. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

TOOLS:
------

Assistant has access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```
Use all tools once before responding. pdf_read will give you a pdf of the publication.
When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""

prompt = PromptTemplate(template= prompt)
tools = [generate_document("BASS"), TavilySearchResults(max_results = 3)]
agent = create_react_agent(llm, tools, prompt)
# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
usr_input = """Please give me the following details regarding the BASS algorithm, just a yes or no. Give me a brief description of how you came to each of these answers.
Clustering Method Categorisation used: Centroid-Based, Hierarchical-Based, Density-Based, Distribution-Based, Unsupervised ML, Supervised ML, GCN, Autoencoder, Ensemble.
Clustering Level: Domain detection, Cell level, Both, or Unclear.
Context-Aware representation: Spatially transformed gene expression matrix, Low dimension with spatial embedding, inaccessible or None.
Input Data: Raw or Normalised
Dataset Resolution: Low-Resolution, High-resolution, Cellular, Segmnetation-Free Subcellular.
Multisample Analysis: Can the algorithm be ran on multiple samples at once?
Notable Datasets: DLPFC Visium, Mouse Hypothalamus MERFISH, Spot-Based Low Resolution, Spot-based high resolution, Imaging-based.
Reference Inclusion: scRNA-seq, Imaging Data, None.
Scalability: Not Scalable (Few Thousand Cells), Scalable (A million cells), Atlas-level (Millions of cells), Unclear.
Overcoming Limitations: Segmentation Errors, Spot Swapping, Sparsity, Batch Effects, Resolution, Rare Cell Types.
Stage Spatial information is added: Early (I.e. Augmented Input Matrix), Late (Joint low dimension embedding), Tailored (neither early nor late).
Metrics Used for evaluation: Adjusted Rand Index (ARI), Normalised Mutual Information (NMI).
Levels of Simulation: Simulation included, Scalability assessment in the simulation, Accuracy assessment in the simulation. 
"""
memory = ChatMessageHistory(session_id="test-session")
agent_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: memory,
    input_messages_key= "input",
    history_messages_key= "chat_history",
)

response = agent_history.invoke({"input": usr_input},
                       config={"configurable": {"session_id": "<foo>"}})

print(response['output'])
