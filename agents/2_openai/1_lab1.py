import os  # Import the os module to interact with the operating system
import asyncio  # Import the asyncio library for asynchronous programming
from dotenv import load_dotenv  # Import load_dotenv to load environment variables from a .env file
from openai import AsyncAzureOpenAI, AsyncOpenAI  # Import the AsyncAzureOpenAI client for Azure OpenAI, AsyncOpenAI for OpenAI
from openai import OpenAIError  # Import OpenAIError to handle errors from the OpenAI API
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled  # Import necessary classes for agent management

# Disable tracing since we are using Azure OpenAI
set_tracing_disabled(disabled = True)

# Load environment variables from a .env file
load_dotenv(override = True)

# Define the main asynchronous function
async def main(instructions, user_prompt):
    try:
        # -------------------------When using AzureOpenAI-----------------------------------------
        # # Create the Async Azure OpenAI client
        # api_key = os.getenv("azure_openai_api_key")  # Get the API key from environment variables
        # api_version = os.getenv("azure_openai_api_version")  # Get the API version from environment variables
        # azure_endpoint = os.getenv("azure_openai_endpoint")  # Get the Azure endpoint from environment variables
        
        # # Check if the required environment variables are srt
        # if not all([api_key, api_version, azure_endpoint]):
        #     raise ValueError("One or more required environment variables are not set.")
        
        # # Initialize the Async Azure OpenAI client with the retrieved credentials
        # client = AsyncAzureOpenAI(azure_endpoint = azure_endpoint, api_version = api_version, api_key = api_key)
        
        # # Configure the agent with Azure OpenAI
        # agent = Agent(name = "ci-assistant",  # Name of the agent
        #               instructions = instructions,  # Instructions for the agent
        #               model = OpenAIChatCompletionsModel(model = "gpt-4o-mini", openai_client = client)  # Specify the model to use
        #               )
        # -------------------------------END-------------------------------------------------------
        
        
        # ----------------------------When using OpenAI--------------------------------------------
        api_key = os.getenv("OPENAI_API_KEY")  # Get the API key from environment variables
        
        # Check if the required environment variables are srt
        if not all([api_key]):
            raise ValueError("One or more required environment variables are not set.")
        
        # Initialize the AsyncOpenAI client with the retrieved credentials
        client = AsyncOpenAI(api_key = api_key)
        
        # Configure the agent with AsyncOpenAI
        agent = Agent(name = "ci-assistant",  # Name of the agent
                      instructions = instructions,  # Instructions for the agent
                      model = OpenAIChatCompletionsModel(model = "gpt-4o-mini", openai_client = client)  # Specify the model to use
                      )
        # ----------------------------------END---------------------------------------------------
        
        # Run the agent with the user prompt
        result = await Runner.run(agent, user_prompt)  # Use the passed user prompt
        
        # Print the final output from the agent
        print(result.final_output)
        
    except OpenAIError as e:  # Handle errors specific to the OpenAI API
        print(f"OpenAI APU Error : {str(e)}")
    except Exception as e:  # Handle any other unexpected errors
        print(f"An unexpected error occurred : {str(e)}")


# Entry point of the program
if __name__ == "__main__":
    # Define the instruction and user prompt
    instructions = "Telling a joke"  # Example instructions
    user_prompt = "Tell a joke about Autonomous AI Agents"  # Example user prompt
    
    # Run the main coroutine using asyncio's event loop, allowing the asynchronous tasks to execute.
    asyncio.run(main(instructions, user_prompt))