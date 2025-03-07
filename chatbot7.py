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
    print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

# Load and chunk contents of the PDF
print("Loading document...")
loader = PyPDFLoader("docs/report.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(pages)
print(f"Document split into {len(all_splits)} chunks")

# Create vector store and add documents
print("Creating embeddings...")
embeddings = OpenAIEmbeddings()
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)

# Create retriever
retriever = vector_store.as_retriever()

# Function to format documents for context
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to classify user input
def classify_input(user_input, llm):
    return "QUESTION"  # Placeholder for now
    ### Implement your input classification logic here ###

# Create LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")  # You can choose the appropriate model

# Main function to run the chatbot
def main():
    # Initialize conversation history
    conversation_history = []
    
    # Create a RAG prompt that includes conversation history
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions based on the provided context. Use only the information in the context to answer questions."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
        ("system", "Context information for this question: {context}")
    ])
    
    print("\nRAG Chatbot initialized. Type 'exit' to end the conversation.\n")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        
        try:
            # Classify the input
            input_type = classify_input(user_input, llm)
            # Uncomment for debugging
            print(f"[Debug] Input classified as: {input_type}")
            
            # Add user message to history
            conversation_history.append(HumanMessage(content=user_input))
            
            # Prepare chat history for the prompt
            chat_history = conversation_history[:-1]  # Exclude the current message
            
            # Process based on input type
            if input_type in ["COMMAND", "STATEMENT"]:
                # Handle non-question inputs directly with the LLM
                response = llm.invoke(
                    [
                        ("system", "You are a helpful assistant that can discuss documents and answer questions."),
                        *chat_history,
                        ("human", user_input)
                    ]
                )
                answer = response.content
            else:
                # For questions, use RAG to retrieve relevant context
                context_docs = retriever.invoke(user_input)
                context = format_docs(context_docs)
                
                # Invoke the RAG chain with conversation history
                response = llm.invoke(
                    rag_prompt.format_messages(
                        chat_history=chat_history,
                        question=user_input,
                        context=context
                    )
                )
                answer = response.content
            
            # Print the response
            print("\nBot:", answer)
            
            # Add assistant message to history
            conversation_history.append(AIMessage(content=answer))
            
            # Limit history length to prevent token overflow
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 