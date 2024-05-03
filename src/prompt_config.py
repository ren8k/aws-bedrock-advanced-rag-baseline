import copy
import re
from typing import Any

import yaml


class PromptConfig:
    def __init__(
        self,
        config_path: str,
        template_path: str,
        query_path: str,
        query_expansion_tempate_path: str = None,
    ) -> None:
        self.config = self._load_config(config_path)
        self.config_org = copy.deepcopy(self.config)
        self.template = self._load_config(template_path)["template"]
        self.query = self._load_config(query_path)["query"]
        self.model_id = self.config.pop("model_id")
        self.prompt = ""
        self.is_stream = self.config.pop("stream")
        if query_expansion_tempate_path:
            self.query_expansion_conf = self._load_config(query_expansion_tempate_path)
            self.prompt_query_expansion = self._format_template(
                self.query_expansion_conf["template"],
                {
                    "n_queries": self.query_expansion_conf["n_queries"],
                    "output_format": self.query_expansion_conf["output_format"],
                    "question": self.query,
                },
            )

    def _load_config(self, config_path: str) -> Any:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def format_prompt(self, args: dict) -> None:
        self._check_args(self.template, args)
        self.prompt = self.template.format(**args)

    def format_message(self, args: dict) -> None:
        if "claude-3" in self.model_id:
            self._format_message_claude3(args)
        elif "command-r-plus" in self.model_id:
            self._format_message_command_r_plus(args)

    def _format_message_claude3(self, args: dict) -> None:
        message = self.config["messages"][0]["content"][0]["text"]
        self._check_args(message, args)
        self.config["messages"][0]["content"][0]["text"] = message.format(**args)

    def _format_message_command_r_plus(self, args: dict) -> None:
        message = self.config["message"]
        self._check_args(message, args)
        self.config["message"] = message.format(**args)

    def _format_template(self, template: str, args: dict) -> None:
        self._check_args(template, args)
        return template.format(**args)

    # https://github.com/taikinman/langrila/blob/main/src/langrila/prompt_template.py#L34
    def _check_args(self, template: str, args: dict) -> None:
        def odd_repeat_pattern(s: str) -> str:
            return f"(?<!{s}){s}(?:{s}{s})*(?!{s})"

        pattern = re.compile(
            odd_repeat_pattern("{") + "[a-zA-Z0-9_]+" + odd_repeat_pattern("}")
        )
        pattern_sub = re.compile("{|}")
        found_args = set([pattern_sub.sub("", m) for m in pattern.findall(template)])
        input_args = set(args.keys())
        if found_args != input_args:
            if found_args:
                _found_args = "{" + ", ".join(found_args) + "}"
            else:
                _found_args = "{}"

            if input_args:
                _input_args = "{" + ", ".join(input_args) + "}"
            else:
                _input_args = "{}"

            raise ValueError(
                f"Template has args {_found_args} but {_input_args} were input"
            )
