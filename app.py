import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import re
from typing import List, Tuple, Union

from langchain.memory import ChatMessageHistory

from src.df_setup import get_column_names
from src.tool_setup import create_tools
from src.template_setup import setup_template
from src.prompt_setup import create_custom_prompt
from src.output_parser_setup import output_parser
from src.llm_chain_setup import create_llm_chain
from src.llm_setup import llm
from src.agent_executor_setup import create_agent_executor
from src.agent_setup import create_custom_llm_single_action_agent

langchain_history = ChatMessageHistory()

with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(
                label="BI ChatGPT", show_copy_button=True, show_label=True
            )
        with gr.Column(scale=1):
            df_table = gr.Dataframe(label="Dataframe", show_label=True)
    with gr.Row():
        with gr.Column(scale=4):
            msg = gr.Textbox(label="Type your question and hit Enter", show_label=True)
        with gr.Column(scale=1):
            clear = gr.ClearButton([msg, chatbot])
            upload_button = gr.UploadButton(
                "Click to upload CSV file", file_count="single"
            )

    def upload_file(file) -> List[Tuple[str, str]]:
        """Handles file upload and returns chat history."""
        file_path = str(file.name)
        chat_history = [("Uploaded a file", "File is uploaded at: " + file_path)]
        return chat_history

    list_of_lists_observation = []

    def list_to_dataframe() -> pd.DataFrame:
        """Converts a list of lists to a DataFrame."""
        if list_of_lists_observation:
            columns = list_of_lists_observation[0]
            data = list_of_lists_observation[1:]
            return pd.DataFrame(data, columns=columns)
        else:
            return pd.DataFrame()

    def run_code_and_save_plot(code: str, filename: str) -> None:
        """Executes code and saves the resulting plot."""
        code = code.replace("`", "")
        try:
            exec(code)
            plt.tight_layout()
            plt.savefig(filename)
        except Exception as e:
            print(f"Error executing code: {e}")

    def parse_chat_history(chat_history: List[Tuple[str, str]]) -> str:
        """Parses the chat history to extract the file path."""
        try:
            parts = chat_history[0][1].split(": ")
            return parts[1]
        except IndexError as e:
            print(f"Error parsing chat history: {e}")
            return ""

    def respond(
        message: str, chat_history: List[Tuple[str, str]]
    ) -> Tuple[str, List[Tuple[str, Union[str, Tuple[str]]]]]:
        """Generates a response based on the user's message and chat history."""
        file_path = parse_chat_history(chat_history)
        if not file_path:
            return "Error: File path not found.", chat_history

        global df
        try:
            df, column_names_str = get_column_names(file_path=file_path)
            column_names_str = str(column_names_str)
            tools = create_tools(df)
            prompt_history = langchain_history.message
            prompt = create_custom_prompt(
                template=setup_template,
                tools=tools,
                column_names_str=column_names_str,
                prompt_history=prompt_history,
            )
            llm_chain = create_llm_chain(llm=llm, prompt=prompt)
            tool_names = [tool.name for tool in tools]
            agent = create_custom_llm_single_action_agent(
                llm_chain=llm_chain, output_parser=output_parser, tool_names=tool_names
            )
            agent_executor = create_agent_executor(agent=agent, tools=tools)
            response = agent_executor(message)
        except Exception as e:
            return f"Error: {e}", chat_history

        observation_string = ""
        for action, observation in response["intermediate_steps"]:
            try:
                regex = r"Thought\s*\d*\s*:(.*?)\nAction\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
                match = re.search(regex, action.log, re.DOTALL)
                if not match:
                    raise ValueError(f"Could not parse LLM output: `{action.log}`")
                thought = match.group(1).strip()
                action_tool = match.group(2)
                action_input = match.group(3)

                global list_of_lists_observation

                if isinstance(observation, str):
                    observation_string = observation
                elif isinstance(observation, pd.DataFrame):
                    df_observation = observation.reset_index()
                    list_of_lists_observation = [
                        df_observation.columns.values.tolist()
                    ] + df_observation.values.tolist()
                    observation_string = df_observation.to_markdown()
                elif isinstance(observation, pd.Series):
                    df_observation = observation.to_frame().reset_index()
                    list_of_lists_observation = [
                        df_observation.columns.values.tolist()
                    ] + df_observation.values.tolist()
                    observation_string = df_observation.to_markdown()
                elif isinstance(observation, plt.Axes):
                    run_code_and_save_plot(action_input, "plot.png")
                    observation_string = "Refer to the plot below."
                else:
                    observation_string = str(observation)
            except Exception as e:
                observation_string = f"Error processing observation: {e}"

        langchain_history.add_user_message(message)
        langchain_history.add_ai_message(thought)
        langchain_history.add_ai_message(action_tool)
        langchain_history.add_ai_message(action_input)
        langchain_history.add_ai_message(observation_string)

        thought = "Thought process:\n" + thought
        observation_string = "Answer:\n" + observation_string
        action_input = "Code used to generate answer:\n" + action_input
        chat_response = thought + "\n\n" + observation_string + "\n\n" + action_input

        chat_history.append([message, chat_response])

        if observation_string == "Answer:\nRefer to plot below.":
            chat_history.append([None, ("plot.png",)])

        return "", chat_history

    upload_button.upload(upload_file, upload_button, chatbot)

    msg.submit(respond, [msg, chatbot], [msg, chatbot]).then(
        list_to_dataframe, outputs=[df_table]
    )

    # TO-DO: edit to respond to dislike
    chatbot.like(respond, [chatbot, chatbot], [msg, chatbot])

demo.launch(share=False)
