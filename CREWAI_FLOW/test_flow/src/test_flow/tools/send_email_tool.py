# In-built packages (Standard Library modules)
import os
from typing import Dict, Type

# External packages
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# Our Own Imports

class send_email_inputs(BaseModel):
    """Schema for sending email inputs"""
    subject : str = Field(..., description = "subject line for the email")
    html_content : str = Field(..., description = "html content of the email")

class SendMail(BaseTool):
    name : str = "Send email via SendGrid"
    description : str = ("This tool is used to send an email using the SendGrid service.")
    args_schema : Type[BaseModel] = send_email_inputs

    def _run(self, subject : str, html_content : str) -> Dict[str, str]:
        try:
            # Initialize the SendGrid client using the API key from the environment
            sg = SendGridAPIClient(api_key = os.getenv("SENDGRID_API_KEY"))
            
            # Set the verified sender email address (must be verified in your SendGrid dashboard)
            from_email = "siddharthwolverine@gmail.com"
            
            # Set the recipient email address (the person who will receive the email)
            to_email = "siddharth13101999singh@gmail.com"
            
            # Create the email message
            message = Mail(from_email = from_email, 
                           to_emails = to_email, 
                           subject = subject, 
                           html_content = html_content)
            
            # Send the email
            sg.send(message)
            
            # Return a success response
            return {"status" : "success"}
        except Exception as e:
            return {"status" : "failed", "error" : str(e)}
