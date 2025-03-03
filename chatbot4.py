import os
import sys
import json
import datetime
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv, find_dotenv

# Define function to get current time
def get_current_time():
    """Get the current time, optionally in a specific timezone"""
    current_time = datetime.datetime.now()

    return f"The current time is {current_time.strftime('%H:%M:%S')} in your local timezone."

# Function definition for the model
FUNCTION_DEFINITIONS = [
    {
        "name": "get_current_time",
        "description": "Get the current time, optionally in a specific timezone"
    }
]

def main():
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    
    # Ensure API key is set
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Initialize the OpenAI chat model with function calling
    model = ChatOpenAI(model="gpt-4o-mini")
    
    # Create a chat prompt template with a system message
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant who responds in a concise and friendly manner. {personality}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Set personality trait
    personality = "You particularly enjoy explaining technical concepts and giving examples. When asked for the time, you are able to retrieve the current time."
    
    # Initialize conversation history
    history = []
    
    print("Starting conversation with OpenAI chat model. Type 'exit' to quit.\n")
    print(f"System personality: {personality}\n")
    print("You can ask for the current time and I'll retrieve it for you!\n")
    
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
            # Invoke the model with function definitions
            response = model.invoke(
                formatted_messages,
                functions=FUNCTION_DEFINITIONS
            )
            
            # Check if the model wants to call a function
            function_call = response.additional_kwargs.get("function_call")
            
            if function_call:
                # Extract function name and arguments
                function_name = function_call["name"]
                function_args = json.loads(function_call["arguments"])
                
                print(f"\n[System] Calling function: {function_name}")
                
                # Execute the function
                if function_name == "get_current_time":
                    result = get_current_time()
                else:
                    result = f"Function {function_name} not implemented"
                
                print(f"[System] Function result: {result}")
                
                # Add the function result to the conversation
                function_message = FunctionMessage(
                    content=result,
                    name=function_name
                )
                
                # Append this to our formatted messages
                formatted_messages.append(function_message)
                
                # Get a new response from the model after the function call
                final_response = model.invoke(formatted_messages)
                
                # Print the response
                print("\nAI:", final_response.content, "\n")
                
                # Add this exchange to history
                history.append(HumanMessage(content=user_input))
                history.append(function_message)
                history.append(AIMessage(content=final_response.content))
                
            else:
                # Just a normal response without function call
                print("\nAI:", response.content, "\n")
                
                # Add this exchange to history
                history.append(HumanMessage(content=user_input))
                history.append(AIMessage(content=response.content))
            
        except Exception as e:
            print(f"Error: {e}")
            
        # Optional: Limit history to prevent token overflow in long conversations
        if len(history) > 12:  # Adjusted to account for function messages
            history = history[-12:]

if __name__ == "__main__":
    main()