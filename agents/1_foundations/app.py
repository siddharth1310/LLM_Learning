# Import necessary libraries
from dotenv import load_dotenv  # For loading environment variables from a .env file
from openai import OpenAI  # For interacting with OpenAI services
import json  # For handling JSON data
import os  # For interacting with the operating system
from pydantic import BaseModel, Field
import requests  # For making HTTP requests
from pypdf import PdfReader  # For reading PDF files
import gradio as gr  # For querying JSON data using JMESPath syntax
from string import Template  # For creating string templates with placeholders

# Load environment variables from a .env file
load_dotenv(override = True)  # This loads the variables defined in the .env file into the environment

# JSON structure for the record_user_details function
record_user_details_json = {
    "name" : "record_user_details",  # The name of the function being called
    
    # Brief overview of the function's purpose
    "description" : "Use this tool to record that a user is interested in being in touch and provided an email address, name, mobile number and notes(optional)", 
    "parameters" : 
        {
            "type" : "object",  # The data type of the parameters (e.g., "object")
            "properties" :  # A collection of key-value pairs representing the function's parameters
                {
                    "email" : 
                        {
                            "type" : "string",  # Expected data type of the parameter
                            "description" : "The email address of this user",  # Explanation of what the parameter represents
                            "default" : "N/A"  # Default value of the parameter
                        }, 
                    "name" : 
                        {
                            "type" : "string",  # Expected data type of the parameter
                            "description" : "The user's name, if they provided it",  # Explanation of what the parameter represents
                            "default" : "N/A"  # Default value of the parameter
                        }, 
                    "mobile_no" : 
                        {
                            "type" : "string",  # Expected data type of the parameter
                            "description" : "The mobile number of this user",  # Explanation of what the parameter represents
                            "default" : "N/A"  # Default value of the parameter
                        }, 
                    "notes" : 
                        {
                            "type" : "string",  # Expected data type of the parameter
                            "description" : "Any additional information about the conversation that's worth recording to give context",  # Explanation of what the parameter represents
                            "default" : "N/A"  # Default value of the parameter
                        }
                }, 
            "required" : ["email"],  # List of parameters that are mandatory for the function to execute
            "additionalProperties" : False  # Indicates whether additional parameters beyond those specified are allowed
        }
}


# JSON structure for the record_unknown_question function
record_unknown_question_json = {
    "name" : "record_unknown_question",  # The name of the function being called
    
    # Brief overview of the function's purpose
    "description" : "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters" : 
        {
            "type" : "object",  # The data type of the parameters (e.g., "object")
            "properties" :  # A collection of key-value pairs representing the function's parameters
                {
                    "question" : 
                        {
                            "type" : "string",  # Expected data type of the parameter
                            "description": "The question that couldn't be answered",  # Explanation of what the parameter represents
                            "default" : "N/A"  # Default value of the parameter
                        }, 
                }, 
            "required" : ["question"],  # List of parameters that are mandatory for the function to execute
            "additionalProperties" : False  # Indicates whether additional parameters beyond those specified are allowed
        }
}


# List of tools to be passed to the LLM
tools = [
    {
        "type" : "function",  # Specifies that this entry is a function tool
        "function" : record_user_details_json  # The JSON structure for the record_user_details function
    }, 
    {
        "type" : "function",  # Specifies that this entry is a function tool
        "function":  record_unknown_question_json  # The JSON structure for the record_unknown_question function
    }
]


system_prompt = """
You are a professional representative of ${name} on their website, dedicated to answering inquiries about ${name}'s career, background, skills and experience. Your goal is to engage users authentically, reflecting ${name}'s voice and professionalism, as if conversing with a potential client or employer.
You have access to a detailed summary of ${name}'s background and LinkedIn profile, which you should leverage to provide informed and relevant responses.

### Tool Usage:
- **Unknown Questions**: If you encounter a question that you cannot answer based on the provided summary or LinkedIn profile, do not respond to the question at all. For example, if a user asks about ${name}'s favorite movie or unrelated personal interests, simply acknowledge that you cannot provide that information. Use your `record_unknown_question` tool to document the question, including the exact wording. This ensures that all user inquiries are tracked for future reference and can help improve responses in subsequent interactions.
- **User Engagement**: If the user shows interest in further discussions or expresses a desire to connect, actively invite them to provide their name, email. mobile number, and any additional notes they may have. Use your `record_user_details` tool to capture this information. If the user does not provide their name, email, or mobile number, use default values.

## Summary:
${summary}

## LinkedIn Profile:
${linkedin}

With this context, please interact with users, ensuring you remain in character as ${name}.

## Guidelines:
1. **Stay in Character**: Respond as if you are ${name}, maintaining their unique tone and style.
2. **Exude Professionalism**: Your responses should always reflect a professional demeanor suitable for potential clients or employers.
3. **Acknowledge Limitations**: If you lack information on a topic, do not respond to the question. Instead utilize the `record_unknown_question` tool to document the inquiry.
4. **Utilize Context**: Reference the provided summary and LinkedIn profile to enrich your responses.
5. **Foster Engagement**: Encourage users to ask more questions and maintain a lively conversation.
6. **Respect User Inquiries**: Treat all question with respect, providing thoughtful and considerate responses.
7. **Encourage Connection**: Actively invite users to share their name, email, mobile number, and any additional notes for further engagement, ensuring to record it using the appropriate tool. If not provided, use default values for the fields.
8. **Limitations on Knowledge**: Do not use any external knowledge, assumptions, or prior training to answer question. Only respond based on the provided context.
"""


# Define the evaluator system prompt for assessing the quality of responses
evaluator_system_prompt = """
You are an evaluator that decides whether a response to a question is acceptable.
You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is acceptable quality.
The Agent is playing the role of ${name} and is representing ${name} on their website.
The Agent has been instructed to be professional and engaging, as if talking to a potential client or future employer who came across the website.

The Agent has been provided with context on ${name} in the form of their summary and LinkedIn details. Here's the information:
## Summary:
${summary}
## LinkedIn Profile:
${linkedin}

With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback.

## Guidelines:
1. **Stay in Character**: Always respond as if you are ${name}, maintaining their tone and style.
2. **Be Professional**: Ensure that your response are professional and suitable for a potential client or employer.
3. **Acknowledge Limitations**: If you do not know the answer to a question, clearly state that you do not have the information.
4. **Use Provided Context**: Utilize the summary and LinkedIn profile to inform your responses and provide relevant information.
5. **Engage with Users**: Encourage further questions and maintain an engaging conversation.
6. **Respect User Queries**: Treat all user inquiries with respect and provide thoughtful responses.
7. **JSON Format**: Response must be in JSON format with strict adherence to the provided output schema.
8. **Enclose JSON**: Enclose JSON response with ```json on its own line, and close it with triple backticks on a new line.
9. **Markdown Formatting**: Values for each JSON key **should use** `markdown` formatting, including emphasis, italics, lists etc.

## Output Schema:
${json_schema}

# Additional Notes:
- Ensure that the evaluation reflects the quality of the Agent's response in relation to the provided context.
- Consider the engagement level of the response and its appropriateness for the intended audience.
"""


# Define the evaluator user prompt for assessing the latest response in a conversation:
evaluator_user_prompt = """Here's the conversation between the User and the Agent:
${history}
Here's the latest message from the User:
${message}
Here's the latest response from the Agent:
${reply}
Please evaluate the response, replying with whether it is acceptable and your feedback.
"""

# Define the Evaluation model to assess the quality of LLM responses
class Evaluation(BaseModel):
    is_acceptable : bool = Field(..., description = "Indicates if the response is of acceptable quality.")
    feedback : str = Field(..., description = "Feedback on the response quality.")


# Define a base model to allow arbitrary types
class BaseArbitraryModel(BaseModel):
    model_config = {"arbitrary_types_allowed" : True}
    model_config["protected_namespaces"] = ()


# Generate the JSON schema for the Evaluation model
evaluation_json_schema = Evaluation.model_json_schema()


class Me:

    def __init__(self, name, linkedIn_path, summary_path):
        self.openai = OpenAI()
        self.name = name

        # Read the LinkedIn profile PDF file to extract text
        reader = PdfReader(linkedIn_path)  # Initialize the PDF reader with the specified file
        self.linkedin = ""  # Initialize an empty string to hold the extracted LinkedIn profile tet

        # Iterate over each page in the PDF document
        for page in reader.pages:
            text = page.extract_text()  # Extract text from the current page
            if text:  # Check if any text was extracted
                self.linkedin = self.linkedin + text  # Concatenate the extracted text to the LinkedIn string

        # Read the LinkedIn summary from a text file
        with open(summary_path, "r", encoding = "utf-8") as f:
            self.summary = f.read()  # Read the entire content of the summary file into the summary variable
    
    
    # Function to send a notification via Pushover
    def push(self, message):
        # Retrieve Pushover API credentials from environment variables
        pushover_user = os.getenv("PUSHOVER_USER")  # Get the Pushover user key
        pushover_token = os.getenv("PUSHOVER_TOKEN")  # Get the Pushover API token
        
        # Define the Pushover API URL for sending messages
        pushover_url = "https://api.pushover.net/1/messages.json"
        
        # Print the message that will be sent
        print(f"Push : {message}")
        
        # Create the payload for the Pushover API request
        payload = {
            "user" : pushover_user,  # The Pushover user key 
            "token" : pushover_token,  # The Pushover API token
            "message" : message  # The message content to be sent
            }
        
        # Send a POST request to the Pushover API to deliver the message
        requests.post(pushover_url, data = payload)

    
    # Function to record user details and send a notification
    def record_user_details(self, email = "N/A", name = "N/A", mobile_no = "N/A", notes = "N/A"):
        # Create a formatted message for the notification
        notification_message = {
            f"New User Interest Notification:\n"
            f"Name: {name}\n"
            f"Email: {email}\n"
            f"Mobile No: {mobile_no}\n"
            f"Notes: {notes}\n"
            f"Please follow up with the user at your earliest convenience."
        }
        
        # Send a notification with the user's details using the push function
        self.push(notification_message)
        
        # Return a confirmation response indicating that the unknown question has been recorded
        return {"recorded" : "ok"}


    # Function to record an unknown question and send a notification
    def record_unknown_question(self, question):
        # Send a notification with the unknown question using the push function
        self.push(f"Recording {question} asked that I couldn't answer")
        
        # Return a confirmation response indicating that the unknown question has been recorded
        return {"recorded" : "ok"}
    
    
    # Function to handle tool calls made by the LLM
    def handle_tool_calls(self, tool_calls):
        """
        This function processes a list of tool calls generated by the LLM.
        Each tool call is represented as a ChatCompletionMessageToolCall object.
        
        Example of tool_calls:
        [
            ChatCompletionMessageToolCall(
                id="call_mnC1KYpiUrlKYEaRqD9U9",
                function=Function(arguments='{"question":"Can you tell me about Nvidia?"}', name="record_unknown_question"),
                type="function"
            )
        ]
        """
        
        # Initialize an empty list to store the results of each tool call
        results = []
        
        # Iterate over each tool call in the provided list
        for tool_call in tool_calls:
            # Extract the name of the function to be called from the tool call object
            tool_name = tool_call.function.name
            
            # Retrieve the arguments for the function, which are stored as a JSON string
            # Convert the JSON string into a python dictionary for easier access
            arguments = json.loads(tool_call.function.arguments)
            
            # Log the tool name and the arguments being passed for debugging purposes
            print(f"Tool called : {tool_name} || , arguments passed : {arguments}", flush = True)
            
            # Attempt to retrieve the function from the global namespace using its name
            # tool = globals().get(tool_name)
            tool = getattr(self, tool_name, None)
            
            # Check if the function was successfully retrieved (i.e., it exists)
            if tool is not None:
                # Validate the arguments against the expected parameters
                expected_params = tool.__code__.co_varnames[:tool.__code__.co_argcount]  # Get expected parameter names
                
                # Exclude 'self' parameter for bound methods
                if expected_params and expected_params[0] == "self":
                    expected_params = expected_params[1:]
                
                print(f"Expected parameters for tool '{tool_name}' : {expected_params}")  # Log this for easier understanding
                
                # Check for missing parameters
                missing_params = [param for param in expected_params if param not in arguments]
                if missing_params:
                    print(f"Missing parameters for {tool_name} : {missing_params}")
                    result = {"error" : f"Missing parameters : {missing_params}"}
                else:
                    # Call the function with unpacked arguments and store the result
                    result = tool(**arguments)
            else:
                # If the function doesn't exist, initialize the result as an error message
                result = {"error" : f"Function '{tool_name}' not found."}
            
            # Append the result of the function call to the results list
            # Each entry includes the role, the content of the result, and the ID of the tool call
            results.append({
                "role" : "tool",  # Indicates that this entry is a tool result
                "content" : json.dumps(result),  # Convert the result to a JSON string for consistency
                "tool_call_id" : tool_call.id  # Include the unique ID of the tool call for reference
                })

        # Return the compiled list of results from all processed tool calls
        return results
    
    
    # Define a function to extract and convert structured output from a response text
    def structured_output(self, response_text):
        # Extract the JSON part from the string
        json_part = response_text.split("```json")[1].split("```")[0].strip()
        
        # Convert the extracted string to a JSON object
        json_object = json.loads(json_part)
        
        # Now json_object is a Python dictionary
        return json_object
    
    
    # Define a function to evaluate the Agent's response
    def evaluate(self, reply, message, history):
        # Construct the messages to be sent to the LLM for evaluation
        messages = [
            {
                "role" : "system",  # Role of the message sender
                "content" : Template(evaluator_system_prompt).substitute(
                    name = "Siddharth Singh",  # Substitute the user's name
                    linkedin = self.linkedin,  # Substitute the LinkedIn profile
                    summary = self.summary,  # Substitute the summary
                    json_schema = evaluation_json_schema,  # Substitute the JSON schema
                ),
            },
            {
                "role" : "user",  # Role of the user
                "content" : Template(evaluator_user_prompt).substitute(
                    history = history,  # Substitute the conversation history
                    message = message,  # Substitute th latest user message
                    reply = reply  # Substitute the latest agent response
                ),
            },
        ]
        
        # Call the LLM to evaluate the response
        response = self.openai.chat.completions.create(model = "gpt-4o-mini", messages = messages)
        
        # Return the structured output from the response
        return self.structured_output(response.choices[0].message.content)
    
    
    def rerun(self, reply, message, history, feedback):
        updated_system_prompt = Template(system_prompt).substitute(
            name = "Siddharth Singh",  # Substitute the user's name
            linkedin = self.linkedin,  # Substitute the LinkedIn profile
            summary = self.summary,  # Substitute the summary
        ) + "\n\n## Previous answer rejected\nYou just tried to reply, but the quality control rejected your reply\n"
        updated_system_prompt = updated_system_prompt + f"## Your attempted answer : \n{reply}\n\n"
        updated_system_prompt = updated_system_prompt + f"## Reason for rejection : \n{feedback}\n\n"
        
        # Construct the messages to be sent to the LLM
        messages = (
            [
                {
                    "role" : "system",  # Role of the message sender
                    "content" : updated_system_prompt  # Updated system prompt with previous response, and the feedback from LLM
                }
            ]
            + history  # Include the previous chat history
            + [{"role" : "user", "content" : message}]  # Add the current user message
        )
        
        # Call the Azure OpenAI chat completion API with the constructed messages
        response = self.openai.chat.completions.create(model = "gpt-4o-mini", messages = messages)
        
        # Return th content of the response from the LLM
        return response.choices[0].message.content
    
    
    # Function to handle chat interactions with the LLM
    def chat(self, message, history):
        # Construct the messages to be sent to the LLM
        messages = (
            [
                {
                    "role" : "system",  # Role of the message sender, indicating this is a system message
                    "content" : Template(system_prompt).substitute(
                        name = "Siddharth Singh",  # Substitute the user's name into the system prompt
                        linkedin = self.linkedin,  # Substitute the LinkedIn profile text into the system prompt
                        summary = self.summary  # Substitute the summary text into the system prompt
                    ),
                }
            ]
            + history  # Include the previous chat history to maintain context
            + [{"role" : "user", "content" : message}]  # Add the current user message to the messages list
        )
        
        done = False  # Initialize a flag to control the loop for processing LLM responses
        while not done:  # Continue processing until the LLM has finished its response
            # Call the LLM with the constructed messages and the tools available for function calls
            response = self.openai.chat.completions.create(model = "gpt-4o-mini", messages = messages, tools = tools)
            
            # Determine the reason for the LLM's response completion
            finish_reason = response.choices[0].finish_reason
            print(f"Finish Reason -> {finish_reason}")  # Log the finish reason for debugging
            
            # If the LLM indicates it wants to call a tool, handle that case
            if finish_reason == "tool_calls":
                # Step 1 - Extract the message from the response, which may contain tool call information
                message = response.choices[0].message
                print(f"Message going inside -> {message}")  # Log the extracted message for debugging
                
                """
                Below is an example of how it may look like -
                Message going inside -> ChatCompletionMessage(content = None, refusal = None, role = "assistant", annotations = [],
                audio = None, function_call = None, tool_calls = [ChatCompletionMessageToolCall(id = "call_mncajkashasklals", 
                function = Function(arguments = '{"question" : "Can you tell me about Nvidia?"}', name = "record_unknown_question"), 
                type = "function")])
                """
                
                # Step 2 - Extract tool_calls details from the 'ChatCompletionMessage'
                tool_calls = message.tool_calls  # Retrieve the list of tool calls requested by the LLM
                print(f"Tool calls -> {tool_calls}")  # Log the tool calls for debugging
                
                """
                Below is an example of how it may look like -
                Message going inside -> [ChatCompletionMessageToolCall(id = "call_mncajkashasklals", 
                function = Function(arguments = '{"question" : "Can you tell me about Nvidia?"}', name = "record_unknown_question"), 
                type = "function")]
                """
                
                # Step 3 - Call the function 'handle_tool_calls' to process the function execution
                results = self.handle_tool_calls(tool_calls)  # Process the tool calls and obtain results
                print(f"Final Result -> {results}")  # Log the final results for debugging
                
                # Append the LLM's message and the results from tool calls to the message list
                messages.append(message)  # Add the LLM's message to the conversation history
                messages.extend(results)  # Add the results from the tool calls to the conversation history
            else:
                done = True  # Exit the loop if no tool calls are made, indicating the LLM has finished processing
        
        # Return the content of the response from the LLM
        LLM_response = response.choices[0].message.content
        print(f"Reply -> {LLM_response}")
        
        # Evaluate the LLM's response
        evaluation = self.evaluate(LLM_response, message, history)
        print(f"Evaluation Result -> {evaluation}")
        
        # Extract token usage details
        token_count = response.usage.total_tokens  # Adjust based on the actual structure of the response
        print(f"Token count -> {token_count}")  # Print the token count for monitoring
        
        # Check if the evaluation indicates the response is acceptable
        if evaluation["is_acceptable"]:
            print("Passed Evaluation - Returning reply")
            return LLM_response  # Return the acceptable reply
        else:
            print("Failed Evaluation - Retrying")
            print(f"Feedback received -> {evaluation['feedback']}")  # Print feedback for debugging
            # new_LLM_response = rerun(LLM_response, message, history, evaluation["feedback"])  # Retry Logic
            # return new_LLM_response  # Return the new reply after retrying
            return LLM_response  # Return the acceptable reply
    

if __name__ == "__main__":
    me = Me(name = "Siddharth Singh", linkedIn_path = "me/personal_linkedIn.pdf", summary_path = "me/summary.txt")
    gr.ChatInterface(me.chat, type = "messages").launch()