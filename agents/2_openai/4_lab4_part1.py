# ----------- Standard and Third-Party Imports -----------
import os  # For accessing OS environment variables (like API keys)
import asyncio  # Manage asynchronous execution for concurrent API calls
from typing import (
    Dict,
    Optional,
)  # For running asynchronous code (lets multiple AI calls happen at once)
from dotenv import (
    load_dotenv,
)  # Loads environment variables from a .env file for configuration
from openai import AsyncOpenAI  # Asynchronous (non-blocking) client for the OpenAI API
from openai import OpenAIError  # Handles errors that come from calling OpenAI
from agents.model_settings import ModelSettings
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    Tool,
    WebSearchTool,
    set_tracing_disabled,
    trace,
    function_tool,
)  # For agent building and tracing
from pydantic import BaseModel, Field
from sendgrid.helpers.mail import (
    Mail,
    Email,
    To,
    Content,
)  # Helper classes from SendGrid to construct emails (sender, recipient, subject, content)
import sendgrid  # Official SendGrid SDK to send mails using SendGrid API
from string import Template  # For creating string templates with placeholders

# Disable tracing (logging or debugging info) from the 'agents' package, unless you want to debug deeply
# set_tracing_disabled(disabled = True)

# Load anything stored in a .env file as environment variables. Usually used for sensitive configuration.
load_dotenv(override=True)


openai_api_key = os.getenv(
    "OPENAI_API_KEY"
)  # Get the API key from environment variables

# Check if the required environment variables are srt
if not all([openai_api_key]):
    raise ValueError("One or more required environment variables are not set.")

# Initialize the AsyncOpenAI client with the retrieved credentials
openai_client = AsyncOpenAI(api_key=openai_api_key)


class WebSearchItem(BaseModel):
    reason: str = Field(
        description="Your reasoning for why this search is important to the query."
    )
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(
        description="A list of web searches to perform to best answer the query."
    )


class ReportData(BaseModel):
    short_summary: str = Field(
        description="A short 2-3 sentence summary of the findings."
    )
    markdown_report: str = Field(description="The final report")
    follow_up_questions: list[str] = Field(
        description="Suggested topics to research further"
    )


# ---------------------------------------------------------
# --- Define What Each Agent Should Act Like -------------
# ---------------------------------------------------------
# Here, we are giving "roles" or "personality profiles" to each agent.
# Each is a different kind of sales agent, or a picker who chooses between emails.

agents_config = {
    "Planner Agent": """
        You are a helpful research assistant. Given a query, come up with a set of web searches 
        to perform to best answer the query. Output ${HOW_MANY_SEARCHES} terms to query for.
    """,
    "Search Agent": """
        You are a research assistant. Given a search term, you search the web for that term and 
        produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 
        words. Capture the main points. Write succinctly, no need to have complete sentences or good 
        grammar. This will be consumed by someone synthesizing a report, so it's vital you capture the 
        essence and ignore any fluff. Do not include any additional commentary other than the summary itself.
    """,
    "Writer Agent": """
        You are a senior researcher tasked with writing a cohesive report for a research query. 
        You will be provided with the original query, and some initial research done by a research assistant. 
        You should first come up with an outline for the report that describes the structure and 
        flow of the report. Then, generate the report and return that as your final output. 
        The final output should be in markdown format, and it should be lengthy and detailed. 
        Aim for 5-10 pages of content, at least 1000 words.
    """,
    "Email Agent": """
        You are able to send a nicely formatted HTML email based on a detailed report.
        You will be provided with a detailed report. You should use your tool to send one email, providing the 
        report converted into clean, well presented HTML with an appropriate subject line.
    """,
}


# ----------------- Function Tool: Send HTML Email -------------------
@function_tool
def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
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
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get("SENDGRID_API_KEY"))

    # Set the verified sender email address (must be verified in your SendGrid dashboard)
    from_email = Email(
        email="siddharthwolverine@gmail.com"
    )  # Change this to your verified sender

    # Set the recipient email address (the person who will receive the email)
    to_email = To(
        email="siddharth13101999singh@gmail.com"
    )  # Change this to your recipient

    # Define the email content with the given HTML body and the correct MIME type
    content = Content(mime_type="text/html", content=html_body)

    # Create the email object with sender, recipient, subject, and content
    # Note: Using 'plain_text_content' to store HTML here is a mistake in SendGrid API usage.
    # The correct argument in Mail() is 'html_content' for HTML body, not 'plain_text_content'.
    mail = Mail(
        from_email=from_email, to_emails=to_email, subject=subject, html_content=content
    ).get()

    # Send the email via the SendGrid client
    sg.client.mail.send.post(request_body=mail)

    # Return a success response
    return {"status": "success"}


planner_agent = Agent(
    name="Planner Agent",
    instructions=Template(agents_config["Planner Agent"]).substitute(
        HOW_MANY_SEARCHES=3
    ),
    model=OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=openai_client),
    output_type=WebSearchPlan,
)

search_agent = Agent(
    name="Search Agent",
    instructions=agents_config["Search Agent"],
    tools=[WebSearchTool(search_context_size="low")],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)

writer_agent = Agent(
    name="Writer Agent",
    instructions=agents_config["Writer Agent"],
    model=OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=openai_client),
    output_type=ReportData,
)

email_agent = Agent(
    name="Email Agent",
    instructions=agents_config["Email Agent"],
    tools=[send_html_email],
    model=OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=openai_client),
)


async def plan_searches(query: str):
    """Use the planner_agent to plan which searches to run for the query"""
    print("Planning searches...")
    result = await Runner.run(planner_agent, f"Query : {query}")
    print(f"Will perform {len(result.final_output.searches)} searches")
    print(f"Searches: {result.final_output.searches}")
    print("Finished planning searches")
    return result.final_output


async def search(item: WebSearchItem):
    """Use the search agent to run a web search for each item in the search plan"""
    input = f"Search term: {item.query}\nReason for searching: {item.reason}"
    result = await Runner.run(search_agent, input)
    return result.final_output


async def perform_searches(search_plan: WebSearchPlan):
    """Call search() for each item in the search plan"""
    print("Searching...")
    tasks = [asyncio.create_task(search(item)) for item in search_plan.searches]
    results = await asyncio.gather(*tasks)
    print(f"Got {len(results)} search results")
    print(f"Results: {results}")
    print("Finished searching")
    return results


async def write_report(query: str, search_results: list[str]):
    """Use the writer agent to write a report based on the search results"""
    print("Thinking about report...")
    input = f"Original query: {query}\nSummarized search results : {search_results}"
    print(f"Input Passed to writer agent {input}")
    print("Writing report...")
    result = await Runner.run(writer_agent, input)
    print("Finished writing report")
    print(f"Report summary: {result.final_output.short_summary}")
    return result.final_output


async def send_email(report: ReportData):
    """Use the email agent to send an email with the report"""
    print("Writing email...")
    result = await Runner.run(email_agent, report.markdown_report)
    print("Email sent")
    return report


# ----------- Main Logic Starts Here -------------
async def main(query: str):
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

        with trace("Research trace"):
            print("Starting research...")
            search_plan = await plan_searches(query)
            search_results = await perform_searches(search_plan)
            report = await write_report(query, search_results)
            await send_email(report)
            print("Hooray!")

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
    query = "Latest AI Agent frameworks in 2025"
    # Execute the main asynchronous workflow with all necessary configuration
    asyncio.run(main(query))


[
    'In 2025, several AI agent frameworks have been introduced, enhancing the development and deployment of intelligent systems. Dalet\'s "Dalia" integrates agentic AI across its media ecosystem, streamlining workflows like ingest, production, and archiving. Cisco\'s "Connected Intelligence" suite in Webex incorporates AI agents for tasks such as note-taking and scheduling, aiming to boost productivity. OutSystems\' "Agent Workbench" offers a low-code platform for enterprises to implement agentic AI, facilitating automation and modernization of legacy systems. ([tvtechnology.com](https://www.tvtechnology.com/news/dalet-introduces-dalia-agentic-ai-across-entire-ecosystem?utm_source=openai))\n\nIn academia, frameworks like "Agent Lightning" and "AgentScope 1.0" have been developed. "Agent Lightning" enables reinforcement learning-based training of large language model agents, promoting flexibility and extensibility. "AgentScope 1.0" provides a developer-centric environment for building agentic applications, supporting tool-based agent-environment interactions. Additionally, "InfiAgent" introduces a self-evolving pyramid agent framework designed for infinite scenarios, enhancing scalability and adaptability. ([arxiv.org](https://arxiv.org/abs/2508.03680?utm_source=openai))\n\nThe Model Context Protocol (MCP), introduced by Anthropic in November 2024, has gained traction in 2025. MCP standardizes AI system integration with external tools and data sources, with major AI providers like OpenAI and Google DeepMind adopting it. This protocol aims to simplify development in multi-model environments and enhance interoperability. ([en.wikipedia.org](https://en.wikipedia.org/wiki/Model_Context_Protocol?utm_source=openai))\n\nThese developments reflect a growing emphasis on agentic AI frameworks in 2025, aiming to improve automation, scalability, and adaptability across various industries. ',
    'In 2025, several companies have introduced innovative AI agent frameworks:\n\n- **Dalet** unveiled "Dalia," an Agentic AI solution integrated across its media ecosystem, offering a unified, natural-language interface for tasks like asset management and live ingests. ([tvtechnology.com](https://www.tvtechnology.com/news/dalet-introduces-dalia-agentic-ai-across-entire-ecosystem?utm_source=openai))\n\n- **Cisco** introduced "Connected Intelligence" within its Webex platform, featuring AI agents for tasks such as note-taking, polling, and scheduling, aiming to enhance collaboration workflows. ([techradar.com](https://www.techradar.com/pro/cisco-goes-all-in-on-agents-and-it-could-mean-big-changes-in-your-workplace?utm_source=openai))\n\n- **Verdent AI**, founded by Zhijie Chen, launched a platform designed to manage complex software development through autonomous coding agents, focusing on large-scale planning and coordination. ([techradar.com](https://www.techradar.com/pro/like-spock-supporting-captain-kirk-tiktoks-former-head-of-algo-reveals-platform-that-can-handle-hordes-of-coding-ai-agents-at-scale?utm_source=openai))\n\n- **OutSystems** released Agent Workbench, a low-code platform enabling enterprises to develop and coordinate intelligent agents across various workflows and data sources. ([techradar.com](https://www.techradar.com/pro/outsystems-agent-workbench-reaches-general-availability-helping-enterprises-streamline-operations-through-agentic-ai?utm_source=openai))\n\n- **Okta** hosted Oktane 2025, emphasizing secure AI integration through identity management, introducing the "Identity Security Fabric" and open standards like Cross App Access (XAA). ([techradar.com](https://www.techradar.com/pro/live/oktane-2025-all-the-news-and-updates-as-they-happen?utm_source=openai))\n\n- **H Company** launched Runner H, an agentic-API platform, and Surfer-H-CLI, an open-source Chrome agent for browser-based automation, aiming to automate complex tasks and enhance productivity. ([en.wikipedia.org](https://en.wikipedia.org/wiki/H_%28company%29?utm_source=openai))\n\n- **Pegasystems** introduced the Agentic Process Fabric, designed to manage and govern AI agents across various systems and tasks, enabling seamless integration of agents, apps, systems, and data. ([en.wikipedia.org](https://en.wikipedia.org/wiki/Pegasystems?utm_source=openai))\n\n- **Artisan AI**, founded in 2023, builds specialized AI agents known as "Artisans" for business automation, focusing on automating complex tasks to enhance productivity. ([en.wikipedia.org](https://en.wikipedia.org/wiki/Artisan_AI?utm_source=openai))\n\nAdditionally, academic advancements include:\n\n- **Agent Lightning**, a flexible framework enabling reinforcement learning-based training of large language model agents, allowing seamless integration with existing agents. ([arxiv.org](https://arxiv.org/abs/2508.03680?utm_source=openai))\n\n- **InfiAgent**, a self-evolving pyramid agent framework designed for infinite scenarios, introducing mechanisms like agent-as-a-tool and agent self-evolution for versatile multi-agent systems. ([arxiv.org](https://arxiv.org/abs/2509.22502?utm_source=openai))\n\n- **AutoAgent**, a fully-automated, zero-code framework for large language model agents, enabling users to create and deploy agents through natural language alone. ([arxiv.org](https://arxiv.org/abs/2502.05957?utm_source=openai))\n\nThese developments reflect a growing emphasis on autonomous AI agents across various industries, aiming to streamline operations and enhance productivity. ',
    "In 2025, AI agent frameworks are evolving to enhance specialization, collaboration, and accessibility. There's a shift towards task-specific agents tailored for domains like healthcare, legal, and marketing, improving accuracy and efficiency. The rise of multi-agent systems enables agents to collaborate, leading to more robust solutions for complex tasks. Additionally, advancements in memory capabilities allow agents to retain context over time, enhancing their ability to handle long-term goals. ([agentiqo.info](https://agentiqo.info/ai-agent-trends-2025/?utm_source=openai))\n\nOpen-source and self-hosted agents are gaining traction, offering organizations greater control over data privacy and customization. Platforms like AutoGPT and Open Interpreter facilitate on-premise hosting and fine-tuning based on internal data, addressing security and compliance concerns. ([agentiqo.info](https://agentiqo.info/ai-agent-trends-2025/?utm_source=openai)) Major tech companies are investing in AI agent platforms to streamline development and deployment. Microsoft's support for the Model Context Protocol (MCP) aims to standardize agent interactions, while AWS's Bedrock AgentCore provides a suite of modular services for scalable AI agent development. ([reuters.com](https://www.reuters.com/business/microsoft-wants-ai-agents-work-together-remember-things-2025-05-19/?utm_source=openai))\n\nThe integration of NoSQL databases is becoming crucial for AI agents, offering flexibility and scalability to manage real-time data and maintain contextual memory. This approach supports the dynamic and adaptive nature of modern AI agents. ([itpro.com](https://www.itpro.com/technology/artificial-intelligence/from-real-time-to-reasoning-why-nosql-is-core-to-agentic-ai?utm_source=openai)) Furthermore, the development of frameworks like Agent Lightning and AutoAgent enables reinforcement learning-based training and zero-code agent creation, making AI agent development more accessible to a broader audience. ([arxiv.org](https://arxiv.org/abs/2508.03680?utm_source=openai))\n\n\n## Recent Developments in AI Agent Frameworks:\n- [OutSystems' Agent Workbench reaches general availability, helping enterprises streamline operations through agentic AI](https://www.techradar.com/pro/outsystems-agent-workbench-reaches-general-availability-helping-enterprises-streamline-operations-through-agentic-ai?utm_source=openai)\n- [From real-time to reasoning: Why NoSQL is core to agentic AI](https://www.itpro.com/technology/artificial-intelligence/from-real-time-to-reasoning-why-nosql-is-core-to-agentic-ai?utm_source=openai)\n- [AWS looks to super-charge AI agents with Amazon Bedrock AgentCore](https://www.techradar.com/pro/aws-looks-to-super-charge-ai-agents-with-amazon-bedrock-agentcore?utm_source=openai) ",
]
