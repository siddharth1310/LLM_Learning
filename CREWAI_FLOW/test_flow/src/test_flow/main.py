#!/usr/bin/env python
import json
from string import Template

from pydantic import ValidationError
from crewai.flow import Flow, start, listen
from crewai import LLM
from rich.console import Console
from rich.markdown import Markdown
from os import makedirs

from test_flow.crews.mail_drafter.mail_drafter import MailDrafter
from test_flow.schemas import AudienceLevel, UserInput, GeneratedResponse, EvaluatorResponse
from test_flow.prompt import lecture_creation_user_prompt, lecture_evaluator_prompt

console = Console()

#--------------------------------------------------------------CODE VERSION 1---------------------------------------------------------------#
"""
This example demonstrates a basic CrewAI flow implementation step-by-step.

1. Define a flow class by inheriting from CrewAI's Flow base class.
2. Specify the model you want to use (e.g., "gpt-4o-mini") via a class variable `model`.
3. Use the decorator `@start()` on the function which will act as the entry point of the flow.
   - Normally, you might take user input here or initiate the flow's first action.
4. To chain functions together, use the `@listen(previous_function_name)` decorator
   so the next function runs with access to the previous function's output.
5. CrewAI encourages storing all flow data inside the `self.state` object,
   which is a structured dictionary-like container designed to hold inputs, outputs, and intermediate results.
6. Fields like `name` and `audience_level` inside `self.state` hold user inputs.
7. In this example, the usual start function to get user input is commented out.
   Instead, user inputs are collected *outside* the flow in the `kickoff()` function.
   This is done to avoid a common CLI bug where user input overlaps and becomes invisible because of CrewAI's flow logging.
8. After collecting valid input from the command line, the values are placed into `hello_flow.state`, 
   and then the flow is kicked off by calling `hello_flow.kickoff()`.
9. The flow calls the LLM using the provided inputs, processes its completion,
   stores the generated greeting back into the state, and returns it.
10. The final output is printed to the console.

Why collect input outside the flow?  
Due to terminal input visibility issues caused by CrewAI's logging, collecting inputs outside the flow prevents this display glitch, ensuring the user sees what they type.

This setup provides a good foundation for how flows use state management, function chaining, and external input handling in a robust, beginner-friendly way.
"""

# class HelloFlow(Flow):
#     model = "gpt-4o-mini"
    
#     # In this case, disabling below part as we are taking user input outside the Flow definition
#     # @start()
#     # def get_user_input(self):
#     #     # Name will already be set in the state, so just pass
#     #     pass
    
#     # @listen(get_user_input)
#     @start()
#     def greet_user(self):
#         try:
#             prompt = f"Say hello to {self.state["name"]} and tell them about there selected audience - {self.state["audience_level"]}"
#             input_message = [{"role" : "user", "content" : prompt}]
#             result = completion(model = self.model, messages = input_message)
#             greeting_text = result.choices[0].message.content
#             self.state["greeting"] = greeting_text
#             return greeting_text
#         except Exception as e:
#             return f"Error calling LLM: {e}"

# def kickoff():
#     try:
#         print("\n" + "=" * 60)
#         print("📊 Greetings Page")
#         print("=" * 60 + "\n")

#         input_name = input("👤 Enter your name : ").strip()
#         while not input_name:
#             input_name = input("⚠️  Name cannot be empty. Please enter your name : ").strip()
        
#         while True:
#             audience_input = input("Who is your target audience? (beginner/intermediate/advanced) : ").strip().lower()
#             if audience_input in ["beginner","intermediate", "advanced"]:
#                 break
#             else:
#                 print("Invalid input! Please enter 'beginner', 'intermediate', or 'advanced'.")
        
#         # Start flow execution
        
#         # define flow object
#         hello_flow = HelloFlow()
#         hello_flow.state["name"] = input_name
#         hello_flow.state["audience_level"] = audience_input
        
#         # kickoff flow execution
#         final_output = hello_flow.kickoff()
#         print(f"Final Output : {final_output}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

# if __name__ == "__main__":
#     kickoff()

#--------------------------------------------------------------CODE VERSION 2---------------------------------------------------------------#
"""
### CrewAI Flow with Pydantic Schema and CrewAI LLM Integration: Complete Example and Explanation

This example incorporates a realistic and detailed prompt alongside well-documented Pydantic schemas 
for input validation and structured output. The code is designed for ease of understanding and maintainability.

---

1. **Enum Definition (`AudienceLevel`)**:
   - Enforces strict allowed audience levels: "beginner", "intermediate", "advanced".
   - Helps ensure user input correctness and provides clear intent.

2. **Pydantic Models**:
   - **`UserInput`**: Represents all inputs supplied by the user, including:
     - `name`: The user's name.
     - `audience_level`: The target audience level, validated as an Enum.
     - `topic`: The lecture topic user wants to create.
   - **`GeneratedResponse`**: Models the expected JSON response from the LLM with fields:
     - `greeting_message`: A personalized greeting tailored based on user and audience.
     - `important_points`: Bulleted list of key considerations for addressing the audience.
     - `lecture_content`: A detailed, engaging lecture on the topic provided.

3. **Flow Class (`HelloFlow`)**:
   - **Inherits** from `Flow[UserInput]`, meaning `self.state` conforms to `UserInput`.
   - Uses the `@start()` decorator to denote the entry point method.
   - Constructs a comprehensive prompt including clear instructions and guidelines for the LLM.
   - Uses CrewAI's `LLM` with Pydantic validation to receive structured JSON output.
   - Updates the flow state with parsed, typed data and returns a dictionary representation.
   - Has in-method exception handling for LLM call failures.

4. **Robust Input Collection and Validation (`kickoff`)**:
   - Interactive CLI input with repeated prompts until valid data is entered.
   - Input is wrapped with Pydantic model validation, graceful error reporting.
   - Populates flow state with sanitized, validated data before kickoff.

5. **Prompt Design Highlights**:
   - Clear **instructions** guiding LLM behavior.
   - Emphasis on engaging and professional lecture creation.
   - Request for output **strictly in JSON format** matching the defined schema.
   - Encourage Markdown formatting of output for rich-text rendering.
   - Guidance on tone and style tailored to the audience level.
"""

# Enum to restrict audience levels to predefined, valid options.


# This enforces a structured, validated response and prevents accidental typos or undefined fields.
class HelloFlow(Flow[UserInput]):
    model = "gpt-4o-mini"  # Can be your preferred LLM
    
    @start()
    def lecture_creation(self):
        try:
            # Detailed prompt instructing the LLM how to generate a professional and audience-aware lecture.
            user_prompt = lecture_creation_user_prompt
            
            input_message = [{"role" : "user", "content" : Template(user_prompt).substitute(name = self.state.name,
                                                                                            audience_level = self.state.audience_level.value,
                                                                                            topic = self.state.topic)}]
            
            # Initialize the CrewAI LLM with a Pydantic model to parse JSON responses.
            llm = LLM(model = self.model, response_format = GeneratedResponse)
            
            # Call the LLM with the prompt, expecting a JSON structured response.
            response = llm.call(messages = input_message)
            
            # Parse JSON string response into GeneratedResponse Pydantic model
            parsed_response = GeneratedResponse(**json.loads(response))
            
            # Ensure output directory exists before saving
            makedirs("output", exist_ok = True)
            
            # Save the outline to a file
            with open("output/guide_outline.json", "w") as f:
                json.dump(json.loads(response), f, indent = 2)
            
            # Store parsed fields back in the flow's strongly typed state.
            self.state.greeting_message = parsed_response.greeting_message
            self.state.important_points = parsed_response.important_points
            self.state.lecture_content = parsed_response.lecture_content
        
            return self.state
        except Exception as e:
            error_msg = f"Error calling LLM: {e}"
            self.state.greeting = error_msg
            return error_msg
    
    @listen(lecture_creation)
    def lecture_evaluator(self, state):
        try:
            user_prompt = lecture_evaluator_prompt
            
            # print(f"Passed Input inside lecture_evaluator - {state.important_points}, {state.lecture_content}")
            input_message = [{"role" : "user", "content" : Template(user_prompt).substitute(important_points = state.important_points, lecture_content = state.lecture_content)}]
            
            # Initialize the CrewAI LLM with a Pydantic model to parse JSON responses.
            llm = LLM(model = self.model, response_format = EvaluatorResponse)
            
            # Call the LLM with the prompt, expecting a JSON structured response.
            response = llm.call(messages = input_message)
            
            # Parse JSON string response into GeneratedResponse Pydantic model
            parsed_response = EvaluatorResponse(**json.loads(response))
            
            # Ensure output directory exists before saving
            makedirs("output", exist_ok = True)
            
            # Save the outline to a file
            with open("output/evaluation_result.json", "w") as f:
                json.dump(json.loads(response), f, indent = 2)
            
            self.state.evaluator_score_card = parsed_response.score_card
            
            return self.state
            
        except Exception as e:
            error_msg = f"Error calling LLM: {e}"
            self.state.greeting = error_msg
            return error_msg
    
    @listen(lecture_evaluator)
    def send_email(self, state):
        try:
            print("Inside Send Email Function")
            print(f"STATE CONTAINING - {state.topic} |||||| {state.lecture_content}")
            # Run the content crew for this section
            result = MailDrafter().crew().kickoff(inputs = {
                "topic" : state.topic,
                "lecture_content" : state.lecture_content
            })
            print(f"Result after running crew - 'Mail Drafter' - {result.raw}")
            self.state.mail_drafter_crew_result = result.raw
            return self.state
            
        except Exception as e:
            error_msg = f"Error calling LLM: {e}"
            self.state.greeting = error_msg
            return error_msg


def kickoff():
    try:
        print("\n" + "=" * 60)
        print("📊 Lecture Creation Flow")
        print("=" * 60 + "\n")

        # Collect user name; re-prompt until non-empty.
        input_name = input("👤 Enter your name: ").strip()
        while not input_name:
            input_name = input("⚠️  Name cannot be empty. Please enter your name: ").strip()

        # Collect audience level; re-prompt until matches Enum options.
        while True:
            audience_input = input("Who is your target audience? (beginner/intermediate/advanced): ").strip().lower()
            try:
                audience_level = AudienceLevel(audience_input)
                break
            except ValueError:
                print("Invalid input! Please enter 'beginner', 'intermediate', or 'advanced'.")

        # Collect lecture topic; re-prompt until non-empty.
        topic_name = input("📚 Enter the topic for the lecture: ").strip()
        while not topic_name:
            topic_name = input("⚠️  Topic cannot be empty. Please enter the lecture topic: ").strip()

        # Validate inputs using Pydantic schema.
        try:
            user_input = UserInput(name = input_name, audience_level = audience_level, topic = topic_name)
        except ValidationError as e:
            print("Input validation error:", e)
            return
        
        # Initialize flow and populate state from validated input.
        hello_flow = HelloFlow()
        hello_flow.state.name = user_input.name
        hello_flow.state.audience_level = user_input.audience_level
        hello_flow.state.topic = user_input.topic
        
        # Launch flow execution and capture result. 
        final_output = hello_flow.kickoff()
        print(f"FINAL OUTPUT -> {final_output}")
        # print(f"Final Output Greeting:\n{final_output.greeting_message}\n")

        # print("Lecture Content (rendered markdown):")
        # lecture_md = Markdown(final_output.lecture_content)
        # console.print(lecture_md)
        
        # print("Evaluator Result")
        # print(final_output.evaluator_score_card)
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    kickoff()

# Explanation of Flow[YourSchema]
"""
1. When you define your flow class with a Pydantic schema as a generic parameter (e.g., Flow[YourSchema]), the flow's state becomes an instance of that schema.
2. This means the state object is strongly typed and structured according to your schema, which enforces validation and field definitions.
3. Unlike a plain dictionary, you cannot blindly add new attributes—you can only assign to fields explicitly defined in your schema.
4. This approach provides a structured, validated response, ensuring your flow's data is predictable, type-safe, and aligned with your schema.

In summary:
1. Yes, the state is explicitly associated with your Pydantic schema.
2. You must assign values to defined fields only.
3. This enforces a structured, validated response and prevents accidental typos or undefined fields.
"""