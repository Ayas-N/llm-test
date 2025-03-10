from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.tools.tavily_search import TavilySearchResults
import prompts
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = open(file= 'langsmith_apikey.txt').read()
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = open('tavilyapi.txt').read()
api_key = open('gemini_apikey.txt').read()
llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash", temperature = 0.3, max_tokens = None, api_key = api_key)
embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key = api_key)

agent_prompt = PromptTemplate(template= prompts.data_prompt2)
memory = ChatMessageHistory(session_id="test-session")

def respond(usr_input, filename):
    tools = []
    tools.append(TavilySearchResults(max_results = 5))
    agent = create_react_agent(llm, tools, agent_prompt)
    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: memory,
        input_messages_key= "input",
        history_messages_key= "chat_history",
        )
    response = agent_history.invoke({"input":usr_input,},
                        config={"configurable": {"session_id": "<foo>"}})
    return response

def generate_with_agent(algorithm):
    usr_input = f"""Please summarise {algorithm} the spatial transcriptomics algorithm"""

    llm_answer = respond(usr_input, algorithm)["output"]
    return llm_answer

pdfs = [os.path.splitext(filename)[0] for filename in os.listdir("pdfs")]
for i in range(1,6):
    for algo in pdfs:
        with open(f"sim{i}/search_out/{algo}.csv", "w") as f:
            f.write(generate_with_agent(algo))