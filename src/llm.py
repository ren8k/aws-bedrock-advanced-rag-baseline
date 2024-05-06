import copy
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from prompt_config import PromptConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LLM:
    def __init__(self, region: str, model_id: str, is_stream: bool = False) -> None:
        retry_config = Config(
            region_name=region,
            retries={
                "max_attempts": 10,
                "mode": "standard",
            },
        )
        self.bedrock_runtime = boto3.client(
            "bedrock-runtime", config=retry_config, region_name=region
        )
        self.model_id = model_id
        if is_stream:
            self.generate = self.generate_stream_message
        else:
            self.generate = self.generate_message

    def generate_message(self, body: str) -> None:
        try:
            response = self.bedrock_runtime.invoke_model(
                body=body, modelId=self.model_id
            )
            response_body = json.loads(response.get("body").read())
            generated_text = self._get_generated_text(response_body)
            return generated_text
        except ClientError as err:
            message = err.response["Error"]["Message"]
            logger.error("A client error occurred: %s", message)

    def _get_generated_text(self, response_body: dict) -> Any:
        if "claude-3" in self.model_id:
            return response_body["content"][0]["text"]
        elif "command-r-plus" in self.model_id:
            return response_body["text"]

    def generate_stream_message(self, body: str) -> None:
        try:
            response = self.bedrock_runtime.invoke_model_with_response_stream(
                body=body, modelId=self.model_id
            )
            for event in response.get("body"):
                self._show_generated_stream_text(event)
        except ClientError as err:
            message = err.response["Error"]["Message"]
            logger.error("A client error occurred: %s", message)

    # TODO: split method for each model.
    def _show_generated_stream_text(self, event: dict) -> None:
        if "claude-3" in self.model_id:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk["type"] == "content_block_delta":
                if chunk["delta"]["type"] == "text_delta":
                    print(chunk["delta"]["text"], end="")
            elif chunk["type"] == "message_delta":
                print("\n")
                print(f"\nStop reason: {chunk['delta']['stop_reason']}")
                print(f"Stop sequence: {chunk['delta']['stop_sequence']}")
                print(f"Output tokens: {chunk['usage']['output_tokens']}")

        elif "command-r-plus" in self.model_id:
            # https://docs.cohere.com/docs/streaming
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk["event_type"] == "text-generation":
                print(chunk["text"], end="")
            elif chunk["event_type"] == "stream-end":
                print("\n")
                print(f"\nStop reason: {chunk['finish_reason']}")
                print(
                    f"Output tokens: {chunk['amazon-bedrock-invocationMetrics']['outputTokenCount']}"
                )

    def expand_queries(self, prompt_conf: PromptConfig) -> None:
        prompt_conf.format_message({"prompt": prompt_conf.prompt_query_expansion})
        body = json.dumps(prompt_conf.config)

        for attempt in range(prompt_conf.retries):
            try:
                if "claude-3" in self.model_id:
                    generate_text = "{" + self.generate(body)
                else:
                    generate_text = self.generate(body)
                query_expanded = json.loads(generate_text)
                query_expanded["query_0"] = prompt_conf.query
                return query_expanded
            except json.JSONDecodeError:
                if attempt < prompt_conf.retries - 1:
                    print(
                        f"Failed to decode JSON, retrying... (Attempt {attempt + 1}/{prompt_conf.retries})"
                    )
                    continue
                else:
                    raise Exception("Failed to decode JSON after several retries.")

    @classmethod
    def eval_relevance_parallel(
        cls,
        region: str,
        prompt_conf: PromptConfig,
        prompts_and_contexts: list,
        max_workers: int = 10,
    ) -> list:
        results = []

        def generate_single_message(llm: LLM, prompt_and_context: dict):
            prompt_conf_tmp = copy.deepcopy(prompt_conf)
            prompt_conf_tmp.format_message({"prompt": prompt_and_context["prompt"]})
            body = json.dumps(prompt_conf_tmp.config)
            is_relevant = llm.generate_message(body)

            if is_relevant == "True":
                return prompt_and_context["context"]
            else:
                return None

        llm = cls(region, prompt_conf.model_id)

        with ThreadPoolExecutor(max_workers) as executor:
            futures = {
                executor.submit(
                    generate_single_message, llm, prompt_and_context
                ): prompt_and_context
                for prompt_and_context in prompts_and_contexts
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        return results
