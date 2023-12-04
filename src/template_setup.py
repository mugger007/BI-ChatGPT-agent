# Set up the base template
template = """You are working with a pandas dataframe in Python. The name of the dataframe is 'df'.

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