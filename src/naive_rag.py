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


def main(args: argparse.Namespace) -> None:
    config_llm_rag_path = "../config/llm/claude-3_rag.yaml"
    # config_llm_rag_path = "../config/llm/command-r-plus_basic.yaml"
    template_path = "../config/prompt_template/prompt_rag.yaml"
    query_path = "../config/query/query.yaml"

    retriever = Retriever(args.kb_id, args.region)
    prompt_conf = PromptConfig(config_llm_rag_path, template_path, query_path)
    llm = LLM(args.region, prompt_conf.model_id, prompt_conf.is_stream)

    # Retrieve contexts
    retrieval_results = retriever.retrieve(prompt_conf.query)
    contexts = retriever.get_contexts(retrieval_results)

    # Augument prompt
    prompt_conf.format_prompt({"contexts": contexts, "query": prompt_conf.query})
    prompt_conf.format_message({"prompt": prompt_conf.prompt})
    body = json.dumps(prompt_conf.config)

    # Generate message
    generated_text = llm.generate(body)
    print(generated_text)


if __name__ == "__main__":
    args = get_args()
    main(args)
