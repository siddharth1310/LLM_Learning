# ----------- Standard and Third-Party Imports -----------
import os  # For accessing OS environment variables (like API keys)
import asyncio  # Manage asynchronous execution for concurrent API calls
from typing import Dict, Optional  # For running asynchronous code (lets multiple AI calls happen at once)
from dotenv import load_dotenv  # Loads environment variables from a .env file for configuration
from openai import AsyncOpenAI  # Asynchronous (non-blocking) client for the OpenAI API
from openai import OpenAIError  # Handles errors that come from calling OpenAI
from agents import Agent, GuardrailFunctionOutput, Runner, OpenAIChatCompletionsModel, Tool, input_guardrail, set_tracing_disabled, trace, function_tool  # For agent building and tracing
from pydantic import BaseModel, Field
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
    
    # Email Subject Writer: Crafts subject lines that maximize open rates.
    "Email subject writer" : """
    You are a specialist in crafting compelling subject lines for cold sales emails.

    ### Your task:
    - You will be provided with the full message body of a cold sales email.
    - Your goal is to generate a subject line that is:
    - Engaging and attention-grabbing,
    - Relevant to the email content,
    - Likely to encourage the recipient to open the email,
    - Clear and concise, ideally under 60 characters.

    ### Important notes:
    - Focus on professionalism and relevance to SOC2 compliance and audit preparation.
    - Avoid generic or overly salesy language that might trigger spam filters.
    - The subject should accurately represent the email content without exaggeration.
    """,
    
    # HTML Email Body Converter: Turns text/markdown email body into a polished HTML version.
    "HTML Email body converter" : """
    You are tasked with converting plain text or markdown-formatted email content into a polished HTML email body.

    ### Your task:
    - You will receive an email body written in plain text or with markdown formatting.
    - Convert this content into well-structured HTML that includes:
    - Clear, readable formatting,
    - Simple and clean layout suitable for email clients,
    - Appropriate use of headings, paragraphs, bullet points, and clickable links if present,
    - Styles that improve readability and engagement without excessive complexity.

    ### Important notes:
    - Ensure the HTML is compatible with common email clients and renders properly.
    - Maintain all the content from the original text without omission.
    - The design should be professional, visually balanced, and compelling to the reader.
    """,
    
    # Email Manager: Controls the process of subject generation, HTML conversion, and sending.
    "Email Manager" : """
    You act as the Email Manager responsible for preparing and sending sales emails efficiently and professionally.
    
    ### Your workflow:
    1. You receive a plain text email body that needs to be sent.
    2. First, call the `subject_writer` tool to generate a concise and engaging email subject based on the email body.
    3. Next, call the `html_converter` tool to transform the text email body into a clean and well-formatted HTML version.
    4. Finally, call the `send_html_email` tool with the subject and the HTML body to send out the email.

    ### Important notes:
    - You must strictly follow this order: subject generation → HTML conversion → sending.
    - Do not alter the email content manually; use the provided tools for each step.
    - Confirm that each step completes successfully before moving on to the next.
    - Your role is to orchestrate these tools seamlessly to ensure professional email delivery.
    """    
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


# ----------------- Function Tool: Send HTML Email -------------------
@function_tool
def send_html_email(subject : str, html_body : str) -> Dict[str, str]:
    """
    Sends an HTML-formatted email using the SendGrid API.

    <b>*Parameters*</b>
    - subject (str): The subject line of the email to be sent.
    - html_body (str): The HTML content that will be used as the body of the email.

    <b>*Returns*</b>
    - Dict[str, str]: A dictionary indicating the status of the operation (e.g., {"status": "success"}).

    <b>*Logic*</b>
    1. Initialize the SendGrid client using the required API key, retrieved securely from environment variables.
    2. Define and validate the sender's email address (must be verified in SendGrid).
    3. Define the recipient's email address.
    4. Construct the email content using the provided HTML body with MIME type "text/html".
    5. Assemble the Mail object with sender, recipient, subject, and HTML content.
    6. Send the email using SendGrid's Mail API with a POST request.
    7. Return a status message indicating the outcome ("success" if the operation was completed).
    """

    # Initialize the SendGrid client using the API key from the environment
    sg = sendgrid.SendGridAPIClient(api_key = os.environ.get("SENDGRID_API_KEY"))

    # Set the verified sender email address (must be verified in your SendGrid dashboard)
    from_email = Email(email = "siddharthwolverine@gmail.com")  # Change this to your verified sender

    # Set the recipient email address (the person who will receive the email)
    to_email = To(email = "siddharth13101999singh@gmail.com")  # Change this to your recipient

    # Define the email content with the given HTML body and the correct MIME type
    content = Content(mime_type = "text/html", content = html_body)

    # Create the email object with sender, recipient, subject, and content
    # Note: Using 'plain_text_content' to store HTML here is a mistake in SendGrid API usage.
    # The correct argument in Mail() is 'html_content' for HTML body, not 'plain_text_content'.
    mail = Mail(from_email = from_email, to_emails = to_email, subject = subject, html_content = content).get()

    # Send the email via the SendGrid client
    sg.client.mail.send.post(request_body = mail)

    # Return a success response
    return {"status" : "success"}


# 1. Define the output format for the guardrail
# ---------------------------------------------
# Purpose:
# This guardrail checks if the user is including a personal name in their message.
# If a name is detected, the guardrail will trigger and prevent the agent from proceeding.
# This class defines the output format that the name-checking guardrail agent will produce.
# Inherits from BaseModel (from Pydantic), which ensures strong typing and validation.
# It has two fields:
    # is_name_in_message (bool): Will be True if a personal name is detected in the message, False otherwise.
    # name (str): The actual name found in the message, or an empty string if none found.
class NameCheckOutput(BaseModel):
    """
    Defines the expected structure for the output returned by the guardrail agent.
    
    <b>*Attributes*</b>
    - is_name_in_message (bool): Indicates whether a personal name was detected in the input message.
      Defaults to False if no name is found.
    - name (str): The detected personal name found in the message. Defaults to empty string if no name is detected.

    This model ensures that the AI agent’s response can be reliably parsed and validated.
    """
    
    is_name_in_message : bool = Field(default = False, description = "True if a personal name was detected in the input message, else False.")
    name : str = Field(default = "", description = "The detected personal name extracted from the message, or an empty string if none found.")


# -------------------------------------------------------------
# 2. Create a specialized agent to check for personal names 
# -------------------------------------------------------------
# Purpose:
# This Agent is specialized for detecting if a message includes a personal name.
# instructions: Direct it to answer only whether a personal name is present.
# output_type: Ensures its response follows the NameCheckOutput structure.
# model: Specifies which language model to use (here, gpt-4o-mini).
guardrail_agent = Agent(
    name = "Name check",  # Descriptive name for logs/tracing
    instructions = "Check if the user is including someone's personal name in what they want you to do.",
    output_type = NameCheckOutput,  # Ensures response conforms to the model we defined above
    model = "gpt-4o-mini"  # The AI model used for this task (could also use OpenAIChatCompletionsModel)
)


# --------------------------------------------------------
# 3. Implement the guardrail function with async logic
# --------------------------------------------------------
# Purpose:
    # This function is invoked before your main agent (for example, Sales Manager) processes a message.
    # It is decorated with @input_guardrail so your agentic framework treats it as a guardrail for filtering/flagging input.
# What happens in the function:
    # Run the guardrail agent: The function invokes guardrail_agent asynchronously with the input message.
    # Extract result: Reads is_name_in_message from the agent's structured output.
    # Return GuardrailFunctionOutput:
        # output_info: Includes the found name (if any) for logging or further handling.
        # tripwire_triggered: If is_name_in_message is True, the guardrail has been triggered (i.e., a name was found in user input), 
        # and this information can be used to halt, warn, or log as appropriate.
@input_guardrail  # Decorator that marks this function as a guardrail for input validation/filtering
async def guardrail_against_name(ctx, agent, message):
    """
    Guardrail Function: Checks the user's message for personal names by running the `guardrail_agent`.
    
    <b>*Parameters*</b>
    - ctx: The guardrail context (holds conversation history/context if needed).
    - agent: The agent that is about to be run (not used directly here).
    - message: The message/input from the user that needs to be checked.

    <b>*Returns*</b>
    - `GuardrailFunctionOutput`: Contains information about whether a name was found, 
    and if so, triggers a `tripwire` that can halt, flag, or log the event.
    """

    # Run the name-checking agent with the message, passing along the context if needed
    result = await Runner.run(guardrail_agent, message, context = ctx.context)
    
    # Extract whether a name was found in the message
    is_name_in_message = result.final_output.is_name_in_message
    
    # Construct and return the output for the guardrail system:
    # - output_info: Information dictionary containing the found name details (for logging, audit, or response)
    # - tripwire_triggered: Boolean flag; if True, indicates the guardrail should take action (halt, warn, etc.)
    return GuardrailFunctionOutput(output_info = {"found_name" : result.final_output}, tripwire_triggered = is_name_in_message)


# ----------- Main Logic Starts Here -------------
async def main(email_prompt : str, sales_tools_description : str, email_tools_description : str):
    """
    Main asynchronous entry point that sets up agents, converts them into tools,
    creates orchestrator agents, and triggers the complete email generation, selection,
    formatting, and sending workflow.

    <b>Parameters</b>
    - email_prompt (str): The initial message or command initiating the cold sales email generation process.
    - sales_tools_description (str): Descriptive label used when converting sales-related agents into tools.
    - email_tools_description (str): Descriptive label used when converting email formatting and sending agents into tools.

    <b>Returns</b>
    - None: Coordinates asynchronous execution; output is handled by the agents and tools.

    <b>Logic</b>
    1. Securely load OpenAI API key from environment variables.
    2. Instantiate asynchronous OpenAI client with the API key.
    3. Define sales-related agent roles responsible for drafting and selecting emails.
    4. Create Agent instances for these sales roles using predefined prompts.
    5. Convert the sales agents into tools with descriptive metadata.
    6. Define email formatting and delivery agent roles.
    7. Create Agent instances for email subject writing and HTML conversion.
    8. Convert email agents into tools with descriptive metadata.
    9. Add the send_html_email utility function as a callable tool.
    10. Create an Email Manager agent empowered with email-related tools to orchestrate formatting and sending.
    11. Define a handoff list including the Email Manager agent, indicating control after sales manager completes.
    12. Create the Sales Manager agent with sales tools, handoff info, and specialized instructions.
    13. Use a tracing context to optionally monitor execution.
    14. Run the Sales Manager asynchronously, initiating the whole workflow.
    15. Handle OpenAI API and unexpected errors gracefully.
    """
    
    try:
        # -------------------------When using AzureOpenAI-----------------------------------------------------------
        # # Create the Async Azure OpenAI client
        # api_key = os.getenv("azure_openai_api_key")  # Get the API key from environment variables
        # api_version = os.getenv("azure_openai_api_version")  # Get the API version from environment variables
        # azure_endpoint = os.getenv("azure_openai_endpoint")  # Get the Azure endpoint from environment variables
        
        # # Check if the required environment variables are srt
        # if not all([api_key, api_version, azure_endpoint]):
        #     raise ValueError("One or more required environment variables are not set.")
        
        # # Initialize the Async Azure OpenAI client with the retrieved credentials
        # client = AsyncAzureOpenAI(azure_endpoint = azure_endpoint, api_version = api_version, api_key = api_key)

        # -------------------------------END------------------------------------------------------------------------
        
        
        # ----------------------------When using OpenAI-------------------------------------------------------------
        openai_api_key = os.getenv("OPENAI_API_KEY")  # Get the API key from environment variables

        # Check if the required environment variables are srt
        if not all([openai_api_key]):
            raise ValueError("One or more required environment variables are not set.")

        # Initialize the AsyncOpenAI client with the retrieved credentials
        openai_client = AsyncOpenAI(api_key = openai_api_key)
        # -------------------------------END------------------------------------------------------------------------
        
        
        # ----------- Create Sales Agents and Convert to Tools -----------------------------------------------------

        # Sales agents responsible for creating and selecting drafts
        sales_agent_roles = ["Professional Sales Agent", "Engaging Sales Agent", "Busy Sales Agent", "Sales Picker"]
        
        # Instantiate sales-related agents from configuration
        sales_agents = [create_agent(name, agents_config[name], openai_client) for name in sales_agent_roles]
        
        # Convert agents into executable tools, tagging with tool_description
        sales_tools = [create_tool(agent, sales_tools_description) for agent in sales_agents]
        
        # -------------------------------END------------------------------------------------------------------------
        
        
        # ----------- Create Email Agents and Convert to Tools -----------------------------------------------------
        
        # Email agents responsible for subject crafting and HTML conversion
        email_agent_roles = ["Email subject writer", "HTML Email body converter"]
        
        # Create email Agents instances based on role configurations
        email_agents = [create_agent(name, agents_config[name], openai_client) for name in email_agent_roles]
        
        # Convert agents into executable tools, tagging with tool_description
        email_tools = [create_tool(agent, email_tools_description) for agent in email_agents]
        
        # Add the existing function tool responsible for sending the formatted HTML email
        email_tools.append(send_html_email)
        
        # -------------------------------END-------------------------------------------------------------------------------
        
        
        # ----------- Create Orchestrator Agents: Email Manager and Sales Manager -----------------------------------------
        
        # Creating the Email Manager agent that orchestrates the email creation and sending process
        email_manager_agent = Agent(name = "Email Manager", 
                              instructions = agents_config["Email Manager"], 
                              tools = email_tools, 
                              model = OpenAIChatCompletionsModel(model = "gpt-4o-mini", openai_client = openai_client), 
                              handoff_description = "Convert an email to HTML and send it")
        
        # Define a list named 'handoffs' which contains agents to which control can be passed after task completion
        # In this case, 'email_manager_agent' is listed as the handoff agent responsible for formatting and sending emails
        handoff_agents = [email_manager_agent]
        
        # Instantiate the Sales Manager agent with access to all tools and instructions
        # How does Guardrails fit into the big picture?
        # When set as an input_guardrail for another agent (e.g., Sales Manager), this check automatically 
        # runs on user input before the agent acts.
        # If a user's message contains a personal name, tripwire_triggered becomes True, and your application 
        # can respond accordingly (block, warn, redact, etc.).
        # This helps enforce privacy, compliance, or etiquette rules in any automated system.
        sales_manager_agent = Agent(name = "Sales Manager", 
                              instructions = agents_config["Sales Manager"], 
                              tools = sales_tools, 
                              handoffs = handoff_agents, 
                              model = OpenAIChatCompletionsModel(model = "gpt-4o-mini", openai_client = openai_client), 
                              input_guardrails = [guardrail_against_name])
        
        # -------------------------------END--------------------------------------------------------------------------------
        
        
        # ----------- Run the Sales Manager to Start the Workflow ----------------------------------------------------------

        # Use a tracing context labeled "Automated Sales Development Representative (SDR) Workflow" 
        # to track workflow execution steps
        with trace("Automated Sales Development Representative (SDR) Workflow"):
            # Run the sales_manager_agent agent asynchronously with the message
            # It will generate drafts, select the best one, and hand off for sending automatically
            await Runner.run(sales_manager_agent, email_prompt)
            
        # -------------------------------END--------------------------------------------------------------------------------
    
    
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
    Script Entry Point

    This block executes only when the script is run directly (e.g., python my_script.py),
    not when imported as a module.

    Overview:
    - Sets up the initial campaign prompt and descriptive labels for sales and email tools.
    - Launches the asynchronous main workflow to orchestrate the sales email drafting,
      selection, formatting, and sending sequence using agentic AI.

    <b>Execution Steps:</b>
    1. Define 'email_prompt', containing the task for the Sales Manager agent
       (includes email context, recipient, and sender details).
    2. Define 'sales_tools_description' for labeling sales draft/selection tools.
    3. Define 'email_tools_description' for labeling subject writing, HTML formatting, and sending tools.
    4. Use 'asyncio.run' to start the asynchronous main function,
       ensuring proper event loop setup and smooth async execution.
    """

    # The main message/task for the Sales Manager agent to start the campaign
    email_prompt = "Send out a cold sales email addressed to Dear User"

    # Human-readable description for sales drafting and selection tools
    sales_tools_description = "Write a cold sales email"

    # Descriptive label for subject writing, HTML formatting, and email sending tools
    email_tools_description = "Assist in preparing and sending a sales email"

    # Execute the main asynchronous workflow with all necessary configuration
    asyncio.run(main(email_prompt, sales_tools_description, email_tools_description))
