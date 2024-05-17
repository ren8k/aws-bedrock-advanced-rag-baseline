import argparse
import json
import logging

from llm import LLM
from llm_config import LLMConfig
from prompt_config import PromptConfig
from retriever import Retriever
from utils import load_yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a message using Knowledge Bases for Amazon Bedrock."
    )
    parser.add_argument(
        "--kb-id",
        type=str,
        default="XXXXXXXXXX",
        help="The ID of the Knowledge Base to use for retrieval.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="The AWS region where the Knowledge Base is located.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="../config/config_claude-3.yaml",
        help="The path to the configuration YAML file.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    config = load_yaml(args.config_path)

    retriever = Retriever(args.kb_id, args.region)
    prompt_conf = PromptConfig(config["template_rag_path"], config["query_path"])
    llm_conf = LLMConfig(config["config_llm_rag_path"])
    llm = LLM(args.region, llm_conf.model_id, llm_conf.is_stream)

    # Retrieve contexts
    retrieval_results = retriever.retrieve(prompt_conf.query)
    contexts = retriever.get_contexts(retrieval_results)

    # Augument prompt
    prompt_conf.format_prompt({"contexts": contexts, "query": prompt_conf.query})
    llm_conf.format_message(prompt_conf.prompt)
    body = json.dumps(llm_conf.llm_args)

    # Generate message
    generated_text = llm.generate(body)
    logger.info(generated_text)


if __name__ == "__main__":
    args = get_args()
    main(args)
