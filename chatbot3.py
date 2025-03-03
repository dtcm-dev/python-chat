import os
import sys
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv, find_dotenv

def main():
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    
    # Ensure API key is set
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Initialize the OpenAI chat model
    model = ChatOpenAI(model="gpt-4o-mini")
    
    # Create a chat prompt template with a system message
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant who responds in a concise and friendly manner. {personality}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Set personality trait
    personality = "You particularly enjoy explaining technical concepts and giving examples."
    
    # Initialize conversation history
    history = []
    
    print("Starting conversation with OpenAI chat model. Type 'exit' to quit.\n")
    print(f"System personality: {personality}\n")
    
    # Conversation loop
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check for exit command
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Exiting conversation.")
            break
        
        # Format the prompt with history and current input
        formatted_messages = prompt.format_messages(
            personality=personality,
            history=history,
            input=user_input
        )
        
        # Get response from the model
        try:
            response = model.invoke(formatted_messages)
            
            # Print the response
            print("\nAI:", response.content, "\n")
            
            # Add this exchange to history
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=response.content))
            
        except Exception as e:
            print(f"Error: {e}")
            
        # Optional: Limit history to prevent token overflow in long conversations
        if len(history) > 10:  # Keep last 5 exchanges (10 messages)
            history = history[-10:]

if __name__ == "__main__":
    main()