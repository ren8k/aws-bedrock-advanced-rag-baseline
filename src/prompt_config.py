import copy
import re
from typing import Any

import utils


class PromptConfig:
    def __init__(
        self,
        template_path: str,
        query_path: str,
        is_query_expansion: bool = False,
        is_relevance_eval: bool = False,
    ) -> None:
        self.template = utils.load_yaml(template_path)["template"]
        self.query = utils.load_yaml(query_path)["query"]
        self.prompt = ""

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
            self.format_instructions = utils.load_yaml(template_path)[
                "format_instructions"
            ]

    def format_prompt(self, args: dict) -> None:
        self._check_args(self.template, args)
        self.prompt = self.template.format(**args)

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
                    "format_instructions": self.format_instructions,
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
