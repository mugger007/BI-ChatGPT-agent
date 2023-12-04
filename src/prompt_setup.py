from langchain.agents import Tool
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from typing import List, Union

# Set up prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    # The string of column names
    column_names_str: str
    # The list of chat history
    prompt_history: List[Union[HumanMessage, AIMessage]]

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "

        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])

        prompt_history_parsed = ""
        ai_message_counter = 1

        for message in self.prompt_history:
            if message.__class__.__name__ == 'HumanMessage':
                prompt_history_parsed += f'Question: {message.content}\n'
            elif message.__class__.__name__ == 'AIMessage':
                if ai_message_counter == 1:
                    prompt_history_parsed += f'Thought: {message.content}\n'
                    ai_message_counter += 1
                elif ai_message_counter == 2:
                    prompt_history_parsed += f'Action: {message.content}\n'
                    ai_message_counter += 1
                elif ai_message_counter == 3:
                    prompt_history_parsed += f'Action Input: {message.content}\n'
                    ai_message_counter += 1
                else:
                    prompt_history_parsed += f'Observation: {message.content}\n'
                    ai_message_counter = 1                    

        kwargs["prompt_history"] = prompt_history_parsed

        # Add the column names string to kwargs
        kwargs["df_head"] = self.column_names_str

        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]

def create_custom_prompt(template, tools, column_names_str, prompt_history):
    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        column_names_str=column_names_str,
        prompt_history=prompt_history,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )
    return prompt
            

