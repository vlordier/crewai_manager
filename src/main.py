""" Generates a list of agents and tasks configuration from prompts"""

import logging
import os
from typing import Optional

import instructor
from crewai_tools import tools
from dotenv import load_dotenv
from openai import OpenAI

from models import AgentsList, TasksList

# load environment variables
load_dotenv()

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = instructor.patch(OpenAI())

openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")


def generate_clarifications(task_description: str) -> Optional[str]:
    """Generate clarifications for a task description

    Args:
    task_description (str): The task description

    Returns:
    str: The generated clarifications
    """

    # using task description, generate 5 clarifications
    clarification_prompt = f"You are a very talented Business Analyst and you have been asked to write a detailed description of this task: {task_description}. Please provide 5 clarifications that you would need to complete the task. And choose the most relevant to start with. Be concise and only ask one question to clarify the task."

    try:
        clarifications = client.completions.create(
            model=openai_model,
            prompt=clarification_prompt,
            max_tokens=100,
            n=5,
            stop=None,
        )
    except Exception as e:
        logger.error("Failed to generate clarifications: %s", e)
        return None

    clarification_question = clarifications.choices[0].text
    logger.info("Clarification: %s", clarification_question)
    clarification_answer = input(
        "Please provide an answer to the clarification question: "
    )

    prompt_refinement = f"You are a very talented Business Analyst and you have been asked to write a detailed description of this task: {task_description}. Please provide 5 clarifications that you would need to complete the task. And choose the most relevant to start with. Be concise and only ask one question to clarify the task. Clarification: {clarification_question}. Answer: {clarification_answer}. Please provide a refined task description based on the clarification."

    try:
        refinement = client.completions.create(
            model=openai_model, prompt=prompt_refinement, max_tokens=100, n=5, stop=None
        )
    except Exception as e:
        logger.error("Failed to generate clarifications: %s", e)
        return None

    refinement_summary = refinement.choices[0].text

    return refinement_summary


def main() -> None:
    """Main function to generate clarifications for a task description"""

    # the AgentList and TaskList have the following format: (JSON)
    agentslist_json = AgentsList.to_json()
    taskslist_json = TasksList.to_json()

    logger.info("AgentList: %s", agentslist_json)
    logger.info("TaskList: %s", taskslist_json)

    for tool in tools:
        logger.info("Class Name: %s", tool.__name__)
        logger.info("Description: %s", tool.__doc__)
        logger.info("Name: %s", tool.name)
        logger.info("Text description: %s", tool.description)
        logger.info("--------------------")

    # get human input
    task_description = input("Please describe the task: ")

    clarified_task = generate_clarifications(task_description)

    if clarified_task:
        logger.info("Clarified task: %s", clarified_task)
    else:
        logger.error("Failed to generate clarifications")

    logger.info("Done!")


if __name__ == "__main__":
    # run the main function
    main()
