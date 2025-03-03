import os
import sys
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
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
    
    # Get user input
    user_input = input("Enter your prompt: ")
    
    # Create a message
    messages = [HumanMessage(content=user_input)]
    
    # Get response from the model
    try:
        response = model.invoke(messages)
        print("\nResponse:")
        print(response.content)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()