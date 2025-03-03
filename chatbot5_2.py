import sys
import os
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
    
# Ensure API key is set
if "OPENAI_API_KEY" not in os.environ:
    print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

# Load and chunk contents of the PDF
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

# Save retrieved documents for later access
def retrieve_and_save(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    return {"context": formatted_context, "retrieved_docs": retrieved_docs, "question": question}

# Main chain that returns both answer and context
rag_chain = (
    retrieve_and_save
    | RunnableParallel(
        answer=(
            {"context": lambda x: x["context"], "question": lambda x: x["question"]}
            | prompt
            | llm
            | StrOutputParser()
        ),
        context=lambda x: x["context"],
        source_documents=lambda x: x["retrieved_docs"]
    )
)

# Example usage
question = input("Enter question here: ")
result = rag_chain.invoke(question)


print("\n=== CONTEXT USED ===")
print(result["context"])
print("\n=== SOURCE DOCUMENTS ===")
for i, doc in enumerate(result["source_documents"]):
    print(f"\nDocument {i+1}:")
    print(f"Page: {doc.metadata.get('page', 'N/A')}")
    print(f"Source: {doc.metadata.get('source', 'N/A')}")
print("\n=== ANSWER ===")
print(result["answer"])