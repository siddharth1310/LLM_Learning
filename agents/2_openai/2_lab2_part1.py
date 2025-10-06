import os  # Import the os module to interact with the operating system for environment variables
import asyncio  # Import the asyncio library to enable asynchronous programming
from dotenv import load_dotenv  # Import load_dotenv to load environment variables from a .env file
from openai import AsyncAzureOpenAI, AsyncOpenAI  # Import the AsyncAzureOpenAI client to interact with Azure OpenAI services
from openai import OpenAIError  # Import OpenAIError to handle errors that may arise from the OpenAI API
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled  # Import necessary classes for managing agents
from openai.types.responses import ResponseTextDeltaEvent  # Import the response type for handling streaming events
import argparse   # Import argparse for command line argument parsing


# Disable tracing to avoid unnecessary logging while using Azure OpenAI
set_tracing_disabled(disabled = True)

# Load environment variables from a .env file, allowing for configuration without hardcoding
load_dotenv(override = True)

# Define a dictionary to store agent names and their corresponding instructions
agents = {
    "Professional Sales Agent" : "You are a sales agent working for CompanionAI, \
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
You write professional, serious cold emails.",
    "Engaging Sales Agent" : "You are a humorous, engaging sales agent working for CompanionAI, \
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
You write witty, engaging cold emails that are likely to get a response.",
    "Busy Sales Agent" : "You are a busy sales agent working for CompanionAI, \
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
You write concise, to the point cold emails."
}


# Define the main asynchronous function
async def main(agent_name, instructions, user_prompt):
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
        sales_agent = Agent(name = agent_name,  # Name of the agent
                      instructions = instructions,  # Instructions for the agent
                      model = OpenAIChatCompletionsModel(model = "gpt-4o-mini", openai_client = client)  # Specify the model to use
                      )
        # ----------------------------------END---------------------------------------------------
        
        # Run the agent and stream the results based on the user prompt
        result = Runner.run_streamed(sales_agent, user_prompt)
        
        async for event in result.stream_events():  # Asynchronously handle streamed events
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end = "", flush = True)  # Print the response as it streams in
        
    except OpenAIError as e:  # Handle errors specific to the OpenAI API
        print(f"OpenAI APU Error : {str(e)}")
    except Exception as e:  # Handle any other unexpected errors
        print(f"An unexpected error occurred : {str(e)}")


# Entry point of the program
if __name__ == "__main__":
    # Define the agent name and retrieve corresponding instructions
    agent_name = "Professional Sales Agent"
    instructions = agents.get(agent_name)
    user_prompt = "Write a cold sales email"  # Example user prompt for the agent
    
    # Run the main coroutine using asyncio's event loop, allowing the asynchronous tasks to execute.
    asyncio.run(main(agent_name, instructions, user_prompt))
    
    # Set up command line argument passing
    # parser = argparse.ArgumentParser(description = "Run a sales agent with specified instructions.")
    # parser.add_argument("agent_name", type = str, help = "Name of the sales agent to use.")
    # parser.add_argument("user_prompt", type = str, help = "User Prompt for the sales agent.")
    
    # # Parse the command line arguments
    # args = parser.parse_args()
    
    # # Retrieve instructions based on the provided agent name
    # instructions = agents.get(args.agent_name)
    # if instructions is None:
    #     print(f"Error : Agent '{args.agent_name}' not found.")
    #     exit(1)
    
    # # Run the main coroutine using asyncio's event loop, allowing the asynchronous tasks to execute.
    # asyncio.run(main(args.agent_name, instructions, args.user_prompt))