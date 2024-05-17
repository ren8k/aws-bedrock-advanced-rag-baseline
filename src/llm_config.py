import copy
from typing import Any

import utils


class LLMConfig:
    def __init__(self, llm_args_path: str) -> None:
        self.llm_args = utils.load_yaml(llm_args_path)
        self.llm_args_org = copy.deepcopy(self.llm_args)
        self.model_id = self.llm_args.pop("model_id")
        self.is_stream = self.llm_args.pop("stream")

    def format_message(self, prompt: str) -> None:
        if "claude-3" in self.model_id:
            self._format_message_claude3(prompt)
        elif "command-r-plus" in self.model_id:
            self._format_message_command_r_plus(prompt)

    def _format_message_claude3(self, prompt: str) -> None:
        message = self.llm_args["messages"][0]["content"][0]["text"]
        self.llm_args["messages"][0]["content"][0]["text"] = message.format(
            prompt=prompt
        )

    def _format_message_command_r_plus(self, prompt: str) -> None:
        message = self.llm_args["message"]
        self.llm_args["message"] = message.format(prompt=prompt)
