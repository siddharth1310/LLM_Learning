#!/usr/bin/env python
from enum import Enum
import json
from typing import List

from pydantic import BaseModel, ValidationError, Field
from litellm import completion # Or use another LLM helper
from crewai.flow import Flow, start, listen
from crewai import LLM

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


"""
### CrewAI Flow with Pydantic Schema: Complete Example and Explanation

This code demonstrates a structured CrewAI flow using 
Pydantic schemas and Enum for robust input validation and state management.

1. **Enum Definition (AudienceLevel):**
   - Restricts audience levels to allowed options: 'beginner', 'intermediate', 'advanced'.
   - Using Enum prevents invalid audience inputs and improves code clarity.

2. **Pydantic Model (UserInput):**
   - Defines the schema of state held within the flow.
   - Includes fields for `name` (user's name) and `audience_level` (restricted by Enum).
   - Ensures only valid, typed data can be assigned to the flow's state.

3. **Flow Class (HelloFlow):**
   - Inherits from `Flow[UserInput]`, making `self.state` a typed `UserInput` instance.
   - Contains the `@start()` method `greet_user` to act as the flow's entry point.
   - Uses validated state data to construct a message prompt.
   - Calls the LLM for completion and safely stores the output in `self.state.greeting`.
   - Includes exception handling to gracefully handle LLM or runtime errors.

4. **Input Collection & Validation (kickoff function):**
   - Gathers user input (name and audience) with validation loops to ensure correctness.
   - Converts inputs into a Pydantic `UserInput` instance, raising validation errors if any.
   - Assigns validated values to the flow's `state` object fields.
   - Calls `kickoff()` to start the flow and prints the final greeting.

5. **Why This Structure?**
   - Separates user input from flow logic to avoid terminal input visibility bugs caused by CrewAI's logging.
   - Strong typing of flow state prevents accidental misuse and improves maintenance.
   - Clear error messaging guides users when input is invalid or external calls fail.
"""

class AudienceLevel(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"

class UserInput(BaseModel, extra = "allow"):
    name : str = None
    audience_level : AudienceLevel = None  # Use the Enum type here

class GeneratedResponse(BaseModel):
    response : str = Field(..., description = "Response generated giving user a greeting message with audience level notation")
    things_to_take_care_of : List[str] = Field(default_factory = [], description = "list of points that help user to take care of while addressing the audience.")

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

# This enforces a structured, validated response and prevents accidental typos or undefined fields.
class HelloFlow(Flow[UserInput]):
    model = "gpt-4o-mini" # Can be your preferred LLM
    
    # In this case, disabling below part as we are taking user input outside the Flow definition
    # @start()
    # def get_user_input(self):
    #     print("I am inside get user input")
    #     # Name will already be set in the state, so just pass
    #     pass

    # @listen(get_user_input)
    @start()
    def greet_user(self):
        try:
            prompt = f"Say hello to {self.state.name} and tell them about there selected audience - {self.state.audience_level.value} and based on it things to be taken care of"
            input_message = [{"role" : "user", "content" : prompt}]
            
            # result = completion(model = self.model, messages = input_message)
            llm = LLM(model = self.model, response_format = GeneratedResponse)
            
            # Make the LLM call with JSON response format
            response = llm.call(messages = input_message)
            
            # Parse the JSON response
            outline_dict = json.loads(response)
            print(f"OUTLINE DICT {outline_dict}")
            final_result = GeneratedResponse(**outline_dict)
            
            self.state.greeting = final_result.response
            self.state.considerations = final_result.things_to_take_care_of
            
        
            print("After execution state result -> ")
            print(self.state)
            
            return outline_dict
            
            
            # # Ensure greeting_text is string, fallback if needed
            # if not isinstance(greeting_text, str):
            #     greeting_text = str(greeting_text) if greeting_text is not None else "Hello!"
            
            # self.state.greeting = greeting_text
            # return greeting_text
        except Exception as e:
            error_msg = f"Error calling LLM: {e}"
            self.state.greeting = error_msg
            return error_msg


def kickoff():
    try:
        print("\n" + "=" * 60)
        print("📊 Greetings Page")
        print("=" * 60 + "\n")
        
        input_name = input("👤 Enter your name : ").strip()
        while not input_name:
            input_name = input("⚠️  Name cannot be empty. Please enter your name : ").strip()
        
        while True:
            audience_input = input("Who is your target audience? (beginner/intermediate/advanced) : ").strip().lower()
            try:
                audience_level = AudienceLevel(audience_input)
                break
            except ValueError:
                print("Invalid input! Please enter 'beginner', 'intermediate', or 'advanced'.")
        
        try:
            user_input = UserInput(name = input_name, audience_level = audience_level)
        except ValidationError as e:
            print("Input validation error:", e)
            return
        
        # Start flow execution
        
        # define flow object
        hello_flow = HelloFlow()
        hello_flow.state.name = user_input.name
        hello_flow.state.audience_level = user_input.audience_level
        
        # kickoff flow execution
        final_output = hello_flow.kickoff()
        print(f"Final Output : {final_output}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    kickoff()