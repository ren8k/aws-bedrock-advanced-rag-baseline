import json
from typing import Any

import boto3
from botocore.config import Config


class LLM:
    def __init__(self, region: str, is_stream: bool = False) -> None:
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
        if is_stream:
            self.generate = self.generate_stream_message
        else:
            self.generate = self.generate_message

    def generate_message(self, body: str, model_id: str) -> None:
        response = self.bedrock_runtime.invoke_model(body=body, modelId=model_id)
        response_body = json.loads(response.get("body").read())
        generated_text = self._get_generated_text(response_body, model_id)
        return generated_text

    def _get_generated_text(self, response_body: dict, model_id: str) -> Any:
        if "claude-3" in model_id:
            return response_body["content"][0]["text"]
        elif "command-r-plus" in model_id:
            return response_body["text"]

    def generate_stream_message(self, body: str, model_id: str) -> None:
        response = self.bedrock_runtime.invoke_model_with_response_stream(
            body=body, modelId=model_id
        )
        for event in response.get("body"):
            self._show_generated_stream_text(event, model_id)

    # TODO: split method for each model.
    def _show_generated_stream_text(self, event: dict, model_id: str) -> None:
        if "claude-3" in model_id:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk["type"] == "content_block_delta":
                if chunk["delta"]["type"] == "text_delta":
                    print(chunk["delta"]["text"], end="")
            elif chunk["type"] == "message_delta":
                print("\n")
                print(f"\nStop reason: {chunk['delta']['stop_reason']}")
                print(f"Stop sequence: {chunk['delta']['stop_sequence']}")
                print(f"Output tokens: {chunk['usage']['output_tokens']}")

        elif "command-r-plus" in model_id:
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
