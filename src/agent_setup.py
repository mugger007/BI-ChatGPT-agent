from langchain.agents import LLMSingleActionAgent
from typing import Any, List, Tuple, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.manager import Callbacks

class CustomLLMSingleActionAgent(LLMSingleActionAgent):
    """Custom LLM single action agent for planning actions."""

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Plans the next action based on intermediate steps."""
        observation_output = ""
        for observation in intermediate_steps:
            observation_output += str(observation)
            print(observation_output)

        if 'error' in observation_output.lower() or observation_output == "":
            output = self.llm_chain.run(
                intermediate_steps=intermediate_steps,
                stop=self.stop,
                callbacks=callbacks,
                **kwargs,
            )
            return self.output_parser.parse(output)
        else:
            return AgentFinish(
                return_values={"output": observation_output},
                log=str(observation_output)
            )

def create_custom_llm_single_action_agent(llm_chain: Any, output_parser: Any, tool_names: List[str]) -> CustomLLMSingleActionAgent:
    """Creates a custom LLM single action agent."""
    try:
        agent = CustomLLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )
        return agent
    except Exception as e:
        print(f"Error creating custom LLM single action agent: {e}")
        return None