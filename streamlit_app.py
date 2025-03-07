import streamlit as st
import sys
import os
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv, find_dotenv

# Load environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Ensure API key is set
if "OPENAI_API_KEY" not in os.environ:
    st.error("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the vector store and retriever
@st.cache_resource
def initialize_rag():
    # Load and chunk contents of the PDF
    with st.spinner("Loading document..."):
        loader = PyPDFLoader("docs/report.pdf")
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(pages)
        st.info(f"Document split into {len(all_splits)} chunks")

        # Create vector store and add documents
        with st.spinner("Creating embeddings..."):
            embeddings = OpenAIEmbeddings()
            vector_store = InMemoryVectorStore(embeddings)
            vector_store.add_documents(all_splits)
            retriever = vector_store.as_retriever()
    
    return retriever

# Function to format documents for context
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to classify user input
def classify_input(user_input, llm):
    return "QUESTION"  # Placeholder for now
    ### Implement your input classification logic here ###

# Create LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Initialize the RAG system
retriever = initialize_rag()

# Create a RAG prompt that includes conversation history
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based on the provided context. Use only the information in the context to answer questions."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
    ("system", "Context information for this question: {context}")
])

# Streamlit UI
st.title("📚 RAG Chatbot")
st.write("Ask questions about the document!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Classify the input
                input_type = classify_input(prompt, llm)
                
                # Prepare chat history for the prompt
                chat_history = [
                    HumanMessage(content=msg["content"]) if msg["role"] == "user"
                    else AIMessage(content=msg["content"])
                    for msg in st.session_state.messages[:-1]  # Exclude the current message
                ]
                
                # Process based on input type
                if input_type in ["COMMAND", "STATEMENT"]:
                    # Handle non-question inputs directly with the LLM
                    response = llm.invoke(
                        [
                            ("system", "You are a helpful assistant that can discuss documents and answer questions."),
                            *chat_history,
                            ("human", prompt)
                        ]
                    )
                    answer = response.content
                else:
                    # For questions, use RAG to retrieve relevant context
                    context_docs = retriever.invoke(prompt)
                    context = format_docs(context_docs)
                    
                    # Invoke the RAG chain with conversation history
                    response = llm.invoke(
                        rag_prompt.format_messages(
                            chat_history=chat_history,
                            question=prompt,
                            context=context
                        )
                    )
                    answer = response.content
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error: {e}")

# Add a sidebar with information
with st.sidebar:
    st.title("About")
    st.write("""
    This is a RAG (Retrieval-Augmented Generation) chatbot that can answer questions about the loaded document.
    
    The chatbot uses:
    - LangChain for RAG implementation
    - OpenAI's GPT-3.5 for text generation
    - Streamlit for the web interface
    
    Type your questions in the chat input below!
    """) 