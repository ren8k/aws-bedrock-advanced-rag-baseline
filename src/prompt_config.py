import copy
import re
from typing import Any

import utils


class PromptConfig:
    def __init__(
        self,
        llm_args_path: str,
        template_path: str,
        query_path: str,
        is_query_expansion: bool = False,
        is_relevance_eval: bool = False,
    ) -> None:
        self.llm_args = utils.load_yaml(llm_args_path)
        self.llm_args_org = copy.deepcopy(self.llm_args)
        self.template = utils.load_yaml(template_path)["template"]
        self.query = utils.load_yaml(query_path)["query"]
        self.model_id = self.llm_args.pop("model_id")
        self.prompt = ""
        self.is_stream = self.llm_args.pop("stream")

        # Query Expansion
        if is_query_expansion:
            self.query_expansion_conf = utils.load_yaml(template_path)
            self.prompt_query_expansion = self._format_template(
                self.template,
                {
                    "n_queries": self.query_expansion_conf["n_queries"],
                    "output_format": self.query_expansion_conf["output_format"],
                    "question": self.query,
                },
            )
            self.retries = self.query_expansion_conf["retries"]

        if is_relevance_eval:
            self.relevance_eval_conf = utils.load_yaml(template_path)

    def format_prompt(self, args: dict) -> None:
        self._check_args(self.template, args)
        self.prompt = self.template.format(**args)

    def format_message(self, args: dict) -> None:
        if "claude-3" in self.model_id:
            self._format_message_claude3(args)
        elif "command-r-plus" in self.model_id:
            self._format_message_command_r_plus(args)

    def _format_message_claude3(self, args: dict) -> None:
        message = self.llm_args["messages"][0]["content"][0]["text"]
        self._check_args(message, args)
        self.llm_args["messages"][0]["content"][0]["text"] = message.format(**args)

    def _format_message_command_r_plus(self, args: dict) -> None:
        message = self.llm_args["message"]
        self._check_args(message, args)
        self.llm_args["message"] = message.format(**args)

    def _format_template(self, template: str, args: dict) -> None:
        self._check_args(template, args)
        return template.format(**args)

    def create_prompts_for_relevance_eval(self, multi_contexts: list) -> list:
        prompts = []
        for context in multi_contexts:
            prompt = self._format_template(
                self.template,
                {
                    "context": context,
                    "question": self.query,
                    "format_instructions": self.relevance_eval_conf[
                        "format_instructions"
                    ],
                },
            )
            prompts.append({"prompt": prompt, "context": context})
        return prompts

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
