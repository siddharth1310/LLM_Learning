# ----------- Standard and Third-Party Imports -----------
import os  # For accessing OS environment variables (like API keys)
import asyncio  # Manage asynchronous execution for concurrent API calls
from typing import Optional  # For running asynchronous code (lets multiple AI calls happen at once)
from dotenv import load_dotenv  # Loads environment variables from a .env file for configuration
from openai import AsyncOpenAI  # Asynchronous (non-blocking) client for the OpenAI API
from openai import OpenAIError  # Handles errors that come from calling OpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, Tool, set_tracing_disabled, trace, function_tool  # For agent building and tracing
from sendgrid.helpers.mail import Mail, Email, To, Content  # Helper classes from SendGrid to construct emails (sender, recipient, subject, content)
import sendgrid  # Official SendGrid SDK to send mails using SendGrid API

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
    # Professional Sales Agent: Writes formal, authoritative, and trust-building sales emails.
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
    
    # Engaging Sales Agent: Writes witty, fun, and engaging sales emails to catch attention.
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
    
    # Busy Sales Agent: Writes short, direct, and clear emails respecting recipients’ time.
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
    
    # Sales Picker: Selects the best draft email based on customer perspective.
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
    
    # Sales Manager: Orchestrates the entire email process, delegating tasks to tools.
    "Sales Manager" : """
    You are the Sales Manager at CompanionAI, responsible for running a cold sales email campaign in an organized and efficient manner.

    ### Task Overview
    Your role is to coordinate the generation, evaluation, and sending of cold sales emails by utilizing specialized tools at your disposal. You will **not write emails yourself** but delegate writing, selection, and sending exclusively to the provided tools.

    ### Tool Usage Instructions
    - **Draft Generation:** Use the following tools **exactly once each** to generate three distinct cold sales email drafts:
    - `professional_sales_agent`
    - `engaging_sales_agent`
    - `busy_sales_agent`
    
    Wait patiently until all three drafts are completed.

    - **Email Selection:** Use the `sales_picker` tool **once** to evaluate the three drafts and choose the single best email. Do not modify or write any email yourself.
    - **Email Sending:** Use the `send_email` tool **once** to send only the chosen best email. Never send more than one email.

    ### Critical Rules
    - Do **not** write, modify, or choose emails manually.
    - Use the tools **in the proper sequence**: generate drafts → pick the best → send the email.
    - Strictly adhere to calling each tool once to avoid loops or redundant operations.
    - Upon sending the email, end the workflow without further actions.

    ### Communication Style
    - Follow these steps clearly and systematically.
    - Avoid unnecessary commentary or explanations beyond what the tools require.
    - Focus on correctness and efficiency of the entire process.
    """,  
}


# ----------- Helper Function: Agent Creation -------------
def create_agent(name : str, instructions : str, client, tools : Optional[list] = None) -> "Agent":
    """
    Creates an Agent configured with a specific role, an OpenAI client, and optionally a set of tools.

    <b>*Parameters*</b>
    - name (str): The descriptive name of the agent (e.g., "Sales Manager"). This is normalized internally to a lowercase underscore-separated identifier.
    - instructions (str): A prompt or instruction set defining the agent's personality, role, and expected behavior.
    - client: An instance of the asynchronous OpenAI client to enable communication with the language model API.
    - tools (list or None): An optional list of auxiliary tool instances the agent can invoke to perform complex multi-step workflows; defaults to None.

    <b>*Returns*</b>
    - Agent: A fully configured Agent object prepared to process inputs, generate outputs, and if tools are provided, call those tools during execution.

    <b>*Logic*</b>
    1. Normalize the agent's name to a consistent format for logging and internal referencing.
    2. Instantiate the Agent with the provided instructions and assign a robust OpenAI chat model ('gpt-4o-mini') powered by the given client.
    3. Attach the provided tools if any are specified, enabling the agent to utilize them when generating responses.
    4. Return the newly created Agent instance to the caller.

    This design supports creating agents both with and without referencing auxiliary tools seamlessly.
    """
    if tools:
        return Agent(name = name.lower().replace(" ", "_"),  # Agent's friendly name for logging or tracing
            instructions = instructions,  # Instructions defining agent’s specific role
            model = OpenAIChatCompletionsModel(  # Chat model with client integration
                model = "gpt-4o-mini",  # Lightweight yet capable GPT model for chat completions
                openai_client = client,
            ),
            tools = tools,  # Tools exposed to the agent to invoke
        )
    else:
        return Agent(name = name.lower().replace(" ", "_"),  # Agent's friendly name for logging or tracing
            instructions = instructions,  # Instructions defining agent’s specific role
            model = OpenAIChatCompletionsModel(  # Chat model with client integration
                model = "gpt-4o-mini",  # Lightweight yet capable GPT model for chat completions
                openai_client = client,
            )
        )


# ----------- Helper Function: Convert Agent to Tool -------------
def create_tool(agent, tool_description : str) -> "Tool":
    """
    Converts an Agent instance into a Tool, enabling it to be called programmatically.

    <b>Parameters</b>
    - agent (Agent): The Agent object to be wrapped and exposed as a tool.
    - tool_description (str): A textual description detailing the purpose and functionality of the tool, used for metadata and clarity.

    <b>Returns</b>
    - Tool: A Tool object that wraps the agent, providing a formal, callable interface to invoke the agent's capabilities within workflows.

    <b>Logic</b>
    1. Use the agent's existing name as the tool's identifier without modification to maintain consistency.
    2. Assign the provided description to the tool for clearer comprehension and documentation.
    3. Call the agent's built-in `as_tool()` method to perform a standardized and clean conversion, producing a tool that other agents or orchestrators can easily call.
    4. Return the created Tool instance for use in agent workflows.
    """
    return agent.as_tool(
        tool_name = agent.name,  # Maintain the agent’s name as the tool’s identifier
        tool_description = tool_description  # Use the provided, meaningful description for this tool
    )


# ----------------- Function Tool: Send Email -------------------
@function_tool
def send_email(body : str):
    """
    Send out an email with the given body to all sales prospects.

    <b>*Parameters*</b>
    - body (str): The content of the email message to be sent.

    <b>*Returns*</b>
    - dict: A dictionary containing the status of the email sending operation.

    <b>*Logic*</b>
    1. Create a SendGrid client authenticated using the API key stored in environment variables.
    2. Define the sender email address (must be a verified sender in SendGrid).
    3. Define the recipient email address (where the email will be delivered).
    4. Create the email content using the provided body text and set MIME type to plain text.
    5. Construct the Mail object combining sender, recipient, subject, and content.
    6. Send the email via SendGrid's API using a POST request.
    7. Return a success status dictionary indicating the email was sent.
    """

    # Initialize the SendGrid client using the API key from the environment
    sg = sendgrid.SendGridAPIClient(api_key = os.environ.get("SENDGRID_API_KEY"))

    # Set the verified sender email address
    from_email = Email(email = "siddharthwolverine@gmail.com")  # Change this to your verified sender

    # Set the recipient email address
    to_email = To(email = "siddharth13101999singh@gmail.com")  # Change this to your recipient

    # Define the email content with the given text body and MIME type "text/plain"
    content = Content(mime_type = "text/plain", content = body)

    # Create the email message object with sender, recipient, subject, and content
    mail = Mail(from_email = from_email, to_emails = to_email, subject = "Test email", plain_text_content = content).get()

    # Send the email through SendGrid's mail API
    sg.client.mail.send.post(request_body = mail)

    # Return a success response to indicate the email was sent
    return {"status" : "success"}


# ----------- Main Logic Starts Here -------------
async def main(message : str, tool_description : str):
    """
    Main asynchronous entry point that sets up agents, converts them into tools,
    creates the orchestrating Sales Manager agent, and triggers the full email 
    generation, selection, and sending workflow.

    <b>Parameters</b>
    - message (str): The prompt or command initiating the cold sales email generation process.
    - tool_description (str): A descriptive string used to label the tools when converting agents to tools.

    <b>Returns</b>
    - None: This function coordinates asynchronous execution; outputs are handled by the agents and tools themselves.

    <b>Logic</b>
    1. Securely load the OpenAI API key from environment variables to authenticate requests.
    2. Instantiate the asynchronous OpenAI client using the API key.
    3. Define the distinct sales agent roles to be created.
    4. Generate Agent instances for each role using predefined role prompts.
    5. Convert each Agent into a callable Tool, using the provided tool description for metadata.
    6. Append an additional function tool, `send_email`, responsible for email delivery.
    7. Create the Sales Manager Agent with:
       - Its dedicated instructions guiding orchestration of the workflow.
       - Access to all created tools for generating, picking, and sending emails.
       - Specified language model for consistent response generation.
    8. Use a trace context to enable optional monitoring and debugging.
    9. Run the Sales Manager asynchronously, passing the triggering message, thereby executing the generation, selection, and sending of the final email.
    10. Handle and report any exceptions arising from API errors or unexpected failures gracefully.
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
        api_key = os.getenv("OPENAI_API_KEY")  # Get the API key from environment variables

        # Check if the required environment variables are srt
        if not all([api_key]):
            raise ValueError("One or more required environment variables are not set.")

        # Initialize the AsyncOpenAI client with the retrieved credentials
        client = AsyncOpenAI(api_key = api_key)

        # Define agent roles to initialize
        agent_names = ["Professional Sales Agent", "Engaging Sales Agent", "Busy Sales Agent", "Sales Picker"]
        
        # Create Agent instances based on role configurations
        agents = [create_agent(name, agents_config[name], client) for name in agent_names]
        
        # Convert agents into executable tools, tagging with tool_description
        tools = [create_tool(agent, tool_description) for agent in agents]
        
        # Add the send_email function tool for sending the selected email
        tools.append(send_email)
        
        # Instantiate the Sales Manager agent with access to all tools and instructions
        sales_manager = Agent(name = "Sales Manager", instructions = agents_config["Sales Manager"], tools = tools, model = "gpt-4o-mini")

        # Use a trace context for monitoring or debugging the Sales Manager's execution
        with trace("Sales manager"):
            # Run the Sales Manager agent asynchronously with the given message
            # This will execute the whole workflow of generating, selecting, and sending the email
            await Runner.run(sales_manager, message)
        
    # If the OpenAI API gives an error (e.g., invalid key, network error, quota), show it clearly
    except OpenAIError as openai_err:
        # Handle API-specific errors with an explanatory printout
        print(f"OpenAI API Error : {str(openai_err)}")
    except Exception as general_err:
        # Catch all other unexpected exceptions and display an informative message
        print(f"An unexpected error occurred : {str(general_err)}")

# ---------------------------------------------------------

# --- Script Entry Point ---
if __name__ == "__main__":
    """
    This block runs when the script is executed directly (e.g., python my_script.py),
    but not when imported as a module by another script.

    It prepares the initial prompt and tool description, then invokes the main async
    workflow to generate, select, and send a cold sales email.

    <b>Steps:</b>
    1. Define the input message (prompt) for the Sales Manager agent to guide the email campaign.
    2. Provide a brief tool description used when converting agents into callable tools.
    3. Use Python's asyncio event loop runner to execute the asynchronous main function,
       ensuring all async operations are properly managed.
    """

    # Define the initial prompt specifying the email context and recipient style
    message = "Send a cold sales email addressed to 'Dear CEO'"

    # Provide a descriptive label for the tools to clarify their purpose
    tool_description = "Write a cold sales email"

    # Run the asynchronous main function, wiring up the full agentic email workflow
    asyncio.run(main(message, tool_description))

