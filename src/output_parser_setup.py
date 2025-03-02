from langchain.agents import AgentOutputParser
from typing import Union
from langchain.schema import AgentAction, AgentFinish
import re

class CustomOutputParser(AgentOutputParser):
    """Custom output parser for parsing LLM output."""

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        """Parses the LLM output to extract the action and action input."""
        try:
            regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            action = match.group(1).strip()
            action_input = match.group(2)

            return AgentAction(
                tool=action, 
                tool_input=action_input.strip(" ").strip('"'), 
                log=llm_output
            )
        except Exception as e:
            print(f"Error parsing LLM output: {e}")
            return AgentFinish(return_values={"output": "Error"}, log=str(e))

output_parser = CustomOutputParser()