from crewai import Agent, Task
from pydantic import BaseModel


class AgentsList(BaseModel):
    """List of agents

    Attributes:
    agents: list[Agent]
    """

    agents: list[Agent]


class TasksList(BaseModel):
    """List of tasks

    Attributes:
    tasks: list[Task]
    """

    tasks: list[Task]
