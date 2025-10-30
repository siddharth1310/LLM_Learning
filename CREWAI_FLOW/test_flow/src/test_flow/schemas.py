from typing import Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field, conint

class AudienceLevel(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"

# Pydantic model defining the user input schema used as flow state.
class UserInput(BaseModel, extra = "allow"):
    name : str = Field(None, description = "Name of the user requesting the lecture")
    audience_level : AudienceLevel = Field(None, description = "Target audience expertise level")
    topic : str = Field(None, description = "Topic on which the lecture should be created")

# Expected structured JSON response from the LLM.
class GeneratedResponse(BaseModel):
    greeting_message : str = Field(..., description = "Personalized greeting for the user")
    important_points : List[str] = Field(default_factory = list, description = "Key considerations to address this audience")
    lecture_content : str = Field(..., description = "Detailed and engaging lecture content to deliver")

ConstrainedInt = conint(ge = 1, le = 10)

class EvaluatorResponse(BaseModel):
    score_card : Optional[Dict[str, ConstrainedInt]] = Field(None, 
                                                             description = ("A dictionary mapping each important_points "
                                                                            "to its corresponding integer score. "
                                                                            "Each score must be between 1 (lowest) and 10 (highest), "
                                                                            "representing how well the lecture met the key considerations."))

class EmailContent(BaseModel):
    """Composed email content including subject and body"""
    subject : str = Field(description = "Subject line of the email")
    body : str = Field(description = "Body content of the email in HTML format")

class SendEmailResponse(BaseModel):
    """Response after attempting to send an email"""
    status : str = Field(..., description = "Result status of the email sending operation, e.g., 'success' or 'failed'")
    error : Optional[str] = Field(None, description = "Error message if the email sending failed; None if successful")
