from langchain.agents import LLMSingleActionAgent
from typing import Any, List, Tuple, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.manager import Callbacks

class CustomLLMSingleActionAgent(LLMSingleActionAgent):
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        
        # Initialize an empty string to accumulate observations
        observation_output = ""
        
        # Iterate through intermediate steps, convert and accumulate observations to a string
        for observation in intermediate_steps:
            observation_output += str(observation)
            print(observation_output)  # Print the accumulated observation output

        # Check if the accumulated observation output contains an error or is empty
        if 'error' in observation_output.lower() or observation_output == "":
            # If error or empty, run the LLM chain and parse the output
            output = self.llm_chain.run(
                intermediate_steps=intermediate_steps,
                stop=self.stop,
                callbacks=callbacks,
                **kwargs,
            )
            return self.output_parser.parse(output)
        else: 
            # If no errors found in any step, stop iterating and finish with a result
            return AgentFinish(
                return_values={"output": observation_output},
                log=str(observation_output)
            )

def create_custom_llm_single_action_agent(llm_chain, output_parser, tool_names):
    agent = CustomLLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )
    return agent