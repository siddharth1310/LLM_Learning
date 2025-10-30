# In-built packages (Standard Library modules)
from typing import List

# External packages
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from crewai.agents.agent_builder.base_agent import BaseAgent

# Our Own Imports
from test_flow.schemas import EmailContent, SendEmailResponse
from test_flow.tools.send_email_tool import SendMail

@CrewBase
class MailDrafter():
    """MailDrafter crew"""

    agents : List[BaseAgent]
    tasks : List[Task]
    
    @before_kickoff
    def prepare_inputs(self, inputs):
        # Preprocess or modify inputs as needed
        inputs["processed"] = True
        print("Before kickoff:", inputs)
        return inputs

    @after_kickoff
    def summarize_results(self, result):
        # Postprocess results (e.g., logging, analytics)
        print("After kickoff:", result)
        return result

    
    #---------AGENTS---------#   
    @agent
    def email_drafter(self) -> Agent:
        return Agent(config = self.agents_config["email_drafter"], verbose = True)
    
    @agent
    def mail_sender(self) -> Agent:
        return Agent(config = self.agents_config["mail_sender"], tools = [SendMail()], verbose = True)

    #---------TASKS---------#
    @task
    def email_composition(self) -> Task:
        return Task(config = self.tasks_config["email_composition"], output_pydantic = EmailContent)
    
    @task
    def send_email(self) -> Task:
        return Task(config = self.tasks_config["send_email"], output_pydantic = SendEmailResponse)
    
    #---------CREW---------#
    @crew
    def crew(self) -> Crew:
        """Creates the MailDrafter crew"""
        print("RUNNING CREW FROM MAIL DRAFTER")
        return Crew(agents = self.agents, tasks = self.tasks, process = Process.sequential, verbose = True)