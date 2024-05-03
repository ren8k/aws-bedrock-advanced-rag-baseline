import boto3
from botocore.client import Config


class Retriever:
    def __init__(self, kb_id: str, region: str) -> None:
        self.kb_id = kb_id
        bedrock_config = Config(
            connect_timeout=120,
            read_timeout=120,
            retries={"max_attempts": 5, "mode": "standard"},
        )
        self.bedrock_agent_client = boto3.client(
            "bedrock-agent-runtime", config=bedrock_config, region_name=region
        )

    def retrieve(self, query: str, no_of_results: int = 5) -> list:
        response = self.bedrock_agent_client.retrieve(
            retrievalQuery={"text": query},
            knowledgeBaseId=self.kb_id,
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": no_of_results,
                    # "overrideSearchType": "HYBRID",  # optional
                }
            },
        )
        return response["retrievalResults"]

    # fetch context from the response
    def get_contexts(self, retrievalResults: list) -> list:
        contexts = []
        for retrievedResult in retrievalResults:
            contexts.append(retrievedResult["content"]["text"])
        return contexts
