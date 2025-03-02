def get_base_template() -> str:
    """
    Returns the base template for the agent.
    """
    return """You are working with a pandas dataframe in Python. The name of the dataframe is 'df'.

    You should use the tools below to answer the question posed of you.
    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)

    The columns in df are:
    {df_head}

    Begin!

    Previous conversation history:
    {prompt_history}

    New question: {input}
    {agent_scratchpad}"""

def setup_template(tools: str, tool_names: str, df_head: str, prompt_history: str, input_question: str, agent_scratchpad: str) -> str:
    """
    Sets up the template with the provided parameters.

    Args:
        tools (str): The tools available for the agent.
        tool_names (str): The names of the tools.
        df_head (str): The head of the dataframe.
        prompt_history (str): The previous conversation history.
        input_question (str): The new input question.
        agent_scratchpad (str): The agent's scratchpad.

    Returns:
        str: The formatted template.
    """
    try:
        template = get_base_template()
        return template.format(
            tools=tools,
            tool_names=tool_names,
            df_head=df_head,
            prompt_history=prompt_history,
            input=input_question,
            agent_scratchpad=agent_scratchpad
        )
    except KeyError as e:
        raise ValueError(f"Missing key in template formatting: {e}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while setting up the template: {e}")

# Example usage:
# tools = "some tools"
# tool_names = "tool1, tool2"
# df_head = "column1, column2"
# prompt_history = "previous conversation"
# input_question = "What is the sum of column1?"
# agent_scratchpad = "scratchpad content"
# formatted_template = setup_template(tools, tool_names, df_head, prompt_history, input_question, agent_scratchpad)
# print(formatted_template)