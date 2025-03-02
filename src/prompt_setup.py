from langchain.agents import Tool
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from typing import List, Union

class CustomPromptTemplate(BaseChatPromptTemplate):
    """Custom prompt template for formatting messages."""
    template: str
    tools: List[Tool]
    column_names_str: str
    prompt_history: List[Union[HumanMessage, AIMessage]]

    def format_messages(self, **kwargs) -> List[HumanMessage]:
        """Formats messages for the prompt."""
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "

        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])

        prompt_history_parsed = ""
        ai_message_counter = 1

        for message in self.prompt_history:
            if isinstance(message, HumanMessage):
                prompt_history_parsed += f'Question: {message.content}\n'
            elif isinstance(message, AIMessage):
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
        kwargs["df_head"] = self.column_names_str

        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]

def create_custom_prompt(template: str, tools: List[Tool], column_names_str: str, prompt_history: List[Union[HumanMessage, AIMessage]]) -> CustomPromptTemplate:
    """Creates a custom prompt template."""
    try:
        prompt = CustomPromptTemplate(
            template=template,
            tools=tools,
            column_names_str=column_names_str,
            prompt_history=prompt_history,
            input_variables=["input", "intermediate_steps"]
        )
        return prompt
    except Exception as e:
        print(f"Error creating custom prompt: {e}")
        return None


