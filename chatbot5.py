import sys
import os
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # Or other embedding/LLM providers
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
    
# Ensure API key is set
if "OPENAI_API_KEY" not in os.environ:
    print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

# Load and chunk contents of the blog
loader = PyPDFLoader("docs/report.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(pages)

# Create vector store and add documents
embeddings = OpenAIEmbeddings()
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)

# Create retriever
retriever = vector_store.as_retriever()

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

# Create LLM
llm = ChatOpenAI()

# Create RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Example usage
question = input("Enter question here:")
answer = rag_chain.invoke(question)
print(answer)