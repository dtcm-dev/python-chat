import os
import sys
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv, find_dotenv

def main():

    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    
    # Ensure API key is set
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OpenAI API key not found. Please provide it as an argument or set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Initialize the OpenAI chat model
    model = ChatOpenAI(model="gpt-4o-mini")   

    # Initialize conversation history
    messages = []
    
    print("Starting conversation with OpenAI chat model. Type 'exit' to quit.\n")
    
    # Conversation loop
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check for exit command
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Exiting conversation.")
            break
        
        # Add user message to history
        messages.append(HumanMessage(content=user_input))
        
        # Get response from the model
        try:
            response = model.invoke(messages)
            
            # Print the response
            print("\nAI, :", response.content, "\n")
            
            # Add AI response to the conversation history
            messages.append(AIMessage(content=response.content))
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()