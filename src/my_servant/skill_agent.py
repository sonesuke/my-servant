import inspect
import json
from typing import Callable, get_type_hints
import ollama
from prompts import FUNCTION_CALLING_PROMPT


class SkillAgent:
    """Agent that uses Ollama API to call skills."""

    def __init__(self, skills: list[Callable]):
        """Initialize the SkillAgent.

        :param skills: List of skills that the agent can use.
        """
        self.skill_functions = skills
        self.skill_jsons = [self.function_to_json(skill) for skill in skills]

    def function_to_json(self, function: Callable) -> str:
        """Convert a function to JSON.

        :param function: Function to be converted.
        :return: JSON representation of the function.
        """

        TYPE_MAP = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }

        skill_dict = {
            "name": function.__name__,
            "description": function.__doc__,
            "parameters": {},
        }

        signature = inspect.signature(function)
        for name, _ in signature.parameters.items():
            skill_dict["parameters"][name] = {
                "type": TYPE_MAP[get_type_hints(function)[name]],
            }

        return json.dumps(skill_dict)

    def select_skill(self, text: str) -> str:
        """Select the skill based on the user input.

        :param text: User input.
        :return: JSON representation of the skill.
        """
        response = ollama.generate(
            model="suzume-mul",
            prompt=FUNCTION_CALLING_PROMPT.format(
                skills=self.skill_jsons, user_input=text
            ),
            format="json",
            stream=False,
        )
        return response["response"]

    def call_skill_function(self, skill_json: str) -> list[dict[str, str]]:
        """Call the skill function based on the skill JSON.

        :param skill_json: JSON representation of the skill.
        :return: List of results from the skill functions.
        """
        skill_dict = json.loads(skill_json)
        response = []
        for skill in skill_dict["skills"]:
            skill_function = next(
                (
                    skill_function
                    for skill_function in self.skill_functions
                    if skill_function.__name__ == skill["skill"]
                ),
                None,
            )
            response.append(
                {
                    "skill": skill["skill"],
                    "result": skill_function(**skill["skillInput"]),
                }
            )
        return response

    def use(self, text: str) -> list[dict[str, str]]:
        """Use the skill agent to call the skill function based on the user input.

        :param text: User input.
        :return: Result of the skill function.
        """
        skill_json = self.select_skill(text)
        return self.call_skill_function(skill_json)


if __name__ == "__main__":

    def get_weather(city: str) -> str:
        """Get the weather of the city."""

        return f"{city}の天気は晴れです。"

    def get_population(country: str) -> str:
        """Get the population of the country."""
        return f"{country}の人口は1000万人です。"

    skills = [get_weather, get_population]
    agent = SkillAgent(skills)
    print(agent.use("京都市の天気を教えて"))
    print(agent.use("日本国の人口を教えて"))
