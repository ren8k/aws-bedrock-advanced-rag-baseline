import argparse
import json
import logging

from botocore.exceptions import ClientError

from llm import LLM
from prompt_config import PromptConfig
from retriever import Retriever

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a message using Knowledge Bases for Amazon Bedrock."
    )
    parser.add_argument(
        "--kb-id",
        type=str,
        default="APNZCYJTKD",
        help="The ID of the Knowledge Base to use for retrieval.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="The AWS region where the Knowledge Base is located.",
    )
    return parser.parse_args()


def query_expansion(llm: LLM, prompt_conf: PromptConfig) -> None:
    prompt_conf.format_message({"prompt": prompt_conf.prompt_query_expansion})
    body = json.dumps(prompt_conf.config)

    try:
        generate_text = llm.generate(body)
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))

    query_expanded = json.loads(generate_text)
    return query_expanded


def main(args: argparse.Namespace) -> None:
    config_path = "../config/llm/claude-3_cofig.yaml"
    # config_path = "../config/llm/command-r-plus_config.yaml"
    template_path = "../config/prompt_template/prompt_template.yaml"
    query_path = "../config/query/query.yaml"
    query_expansion_tempate_path = "../config/query/query_expansion.yaml"

    prompt_conf = PromptConfig(
        config_path, template_path, query_path, query_expansion_tempate_path
    )
    retriever = Retriever(args.kb_id, args.region)
    llm = LLM(args.region, prompt_conf.model_id, prompt_conf.is_stream)

    # Query Expansion
    query_expanded = query_expansion(llm, prompt_conf)
    print(query_expanded)

    # Retrival


if __name__ == "__main__":
    args = get_args()
    main(args)
