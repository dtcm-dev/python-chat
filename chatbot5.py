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


