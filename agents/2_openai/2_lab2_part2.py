# ----------- Standard and Third-Party Imports -----------
import os  # For accessing OS environment variables (like API keys)
import asyncio  # For running asynchronous code (lets multiple AI calls happen at once)
from dotenv import load_dotenv  # Loads environment variables from a .env file for configuration
from openai import AsyncOpenAI  # Asynchronous (non-blocking) client for the OpenAI API
from openai import OpenAIError  # Handles errors that come from calling OpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, trace  # For agent building and tracing
from openai.types.responses import ResponseTextDeltaEvent # For handling streamed response updates

# Disable tracing (logging or debugging info) from the 'agents' package, unless you want to debug deeply
# set_tracing_disabled(disabled = True)

# Load anything stored in a .env file as environment variables. Usually used for sensitive configuration.
load_dotenv(override = True)

# ---------------------------------------------------------
# --- Define What Each Agent Should Act Like -------------
# ---------------------------------------------------------
# Here, we are giving "roles" or "personality profiles" to each agent.
# Each is a different kind of sales agent, or a picker who chooses between emails.

agents_config = {
    # This agent writes very serious, professional emails
    "Professional Sales Agent" : """
    You are a professional sales agent representing CompanionAI, a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits powered by AI.
    
    ### Role and Style:
    - Write cold sales emails that are formal, professional, and serious.
    - Focus on clear, authoritative language that builds trust and highlights the value of the compliance product.
    - Avoid humor or informal expressions; maintain a respectful and businesslike tone.
    - Tailor emails to appeal to decision-makers concerned with security and regulatory compliance.

    ### Instructions:
    - Your emails should be structured logically, with a strong opening, concise benefits, and a clear call to action.
    - Keep the content strictly relevant to SOC2 compliance and audit preparation solutions.
    """,
    
    # This agent adds humor and engagement to its emails (perhaps more catchy)
    "Engaging Sales Agent" : """
    You are an engaging and humorous sales agent for CompanionAI, a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits powered by AI.

    ### Role and Style:
    - Write cold sales emails that are witty, lively, and designed to capture attention.
    - Use humor and lighthearted language to increase engagement and encourage responses.
    - Maintain professionalism but make the tone approachable and fun.
    - Aim to create rapport and spark curiosity about the compliance product.

    ### Instructions:
    - Craft emails that balance entertainment with clear value propositions.
    - Make the recipient feel intrigued and willing to respond or learn more.
    - Use creative hooks or clever phrasing, but avoid excessive jargon or complexity.
    """,
    
    # This agent is quick and gets straight to the point
    "Busy Sales Agent" : """
    You are a busy sales agent working for CompanionAI, a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits powered by AI.

    ### Role and Style:
    - Write cold sales emails that are concise, direct, and to the point.
    - Focus on brevity and clarity without sacrificing key information.
    - Use straightforward language that respects the recipient’s limited time.
    - Quickly convey the value of the product and clear next steps.

    ### Instructions:
    - Keep emails short, ideally under 100 words.
    - Use bullet points or short sentences to highlight benefits.
    - End with a simple call to action.
    """,
    
    # The picker agent: only picks (does not write) the best email out of several
    "Sales Picker" : """
    You are a decision-maker acting as a customer receiving cold sales emails about a SaaS tool for SOC2 compliance from CompanionAI.

    ### Role:
    - You will be shown multiple cold sales email options.
    - Carefully evaluate the emails, imagining your preferences as the customer.

    ### Instructions:
    - Pick the single cold sales email you are most likely to respond to.
    - Do not provide explanations, commentary, or additional text.
    - Only reply with the selected email text verbatim.
    """,
}


# ----------- Helper Function: Agent Creation -------------
def create_agent(name : str, instructions : str, client) -> "Agent":
    """
    Creates an Agent instance with a specific role and associated OpenAI client.

    <b>*Parameters*</b>
    - name (str): The descriptive name used to identify the agent (e.g., "Professional Sales Agent").
    - instructions (str): The prompt or instructions that specify the agent's personality and behavior.
    - client: The initialized AsyncOpenAI client for model communication.

    <b>*Returns*</b>
    - Agent: An agent object set up with the given instructions and client, ready to generate responses.

    <b>*Logic*</b>
    1. Configure an Agent object with a name and personalized instructions.
    2. Bind the OpenAI chat model ('gpt-4o-mini') and provide the API client for requests.
    3. Return the configured agent for further interaction.
    """
    
    return Agent(name = name,  # Agent name (for logging, identification)
        instructions = instructions,  # The 'role' prompt text
        model = OpenAIChatCompletionsModel(  # Wrapping OpenAI chat model
            model = "gpt-4o-mini",  # Name of the language model (can use more advanced ones later)
            openai_client = client
        )
    )


# ----------- Main Logic Starts Here -------------
async def main(message : str):
    """
    Orchestrates the generation and selection of cold sales emails using multiple agents.

    <b>*Parameters*</b>
    - message (str): The input message or prompt to be sent to each sales agent (e.g., "Write a cold sales email").

    <b>*Returns*</b>
    - None (prints output directly to the terminal/console).

    <b>*Logic*</b>
    1. Securely fetch the OpenAI API key from environment variables.
    2. Initialize the AsyncOpenAI client for communication with the OpenAI API.
    3. Create three sales agent objects, each representing a different writing style.
    4. Create a "sales_picker" agent to choose the best email.
    5. Simultaneously run each sales agent to generate different cold email variants.
    6. Collect and format these emails into a single prompt for the picker agent.
    7. Run the picker agent in streamed mode to select the best email, handling output as it arrives.
    8. Manage and report any errors that occur during API calls or execution.
    """
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

        # -------------------------------END-------------------------------------------------------
        
        # ----------------------------When using OpenAI--------------------------------------------
        # 1. Get the OpenAI API key from the environment. This protects your secret key and keeps it out of your code!
        api_key = os.getenv("OPENAI_API_KEY")  # Looks in your computer's environment variables
        
        # 2. If not found, stop and show an error.
        if not all([api_key]):
            raise ValueError("One or more required environment variables are not set.")

        # 3. With the key, create an async OpenAI client for communication
        client = AsyncOpenAI(api_key = api_key)
        # ----------------------------------END---------------------------------------------------
        
        
        # 4. Define which agent names we want to use from the config (all but the picker)
        agent_names = ["Professional Sales Agent", "Engaging Sales Agent", "Busy Sales Agent"]
        
        # 5. Create agent instances, one for each style
        agents = [create_agent(name, agents_config[name], client) for name in agent_names]
        
        # 6. Create the agent that will pick the "best" email
        sales_picker = create_agent("Sales Picker", agents_config["Sales Picker"], client)

        # 7. Enter a context manager for tracing/logging this part of the logic (for optional debugging)
        with trace("Selection from sales people"):
            # 8. Run all your email-writing agents at the same time 
            # (asyncio.gather lets them all work in parallel)
            results = await asyncio.gather(*[Runner.run(agent, message) for agent in agents])
            
            # 9. Gather the email text (from each agent's result)
            outputs = [result.final_output for result in results]

        # 10. Build a big prompt where all emails are shown as options to the picker agent
        emails = "Cold sales emails:\n\n" + "\n\nEmail:\n\n".join(outputs)

        # 11. Ask the picker agent, via streaming, to choose the best one
        # This streams output for large/slow AI responses for better user experience.
        result = Runner.run_streamed(sales_picker, emails)
        
        # 12. As the picker agent sends its response, display it piece-by-piece:
        async for event in result.stream_events():
            # If this "event" is a chunk of actual response text, print it immediately (no waiting for it to finish)
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end = "", flush = True)
        
    # If the OpenAI API gives an error (e.g., invalid key, network error, quota), show it clearly
    except OpenAIError as e:
        print(f"OpenAI API Error : {str(e)}")
    # Catch any other errors (e.g., coding mistakes, other issues)
    except Exception as e:
        print(f"An unexpected error occurred : {str(e)}")

# ---------------------------------------------------------

# --- This runs if you type: python my_script.py -----------
if __name__ == "__main__":
    # Default message to ask the agents (can be replaced as needed)
    message = "Write a cold sales email"
    # Start the main process; asyncio.run sets up the event loop for async functions
    asyncio.run(main(message))
