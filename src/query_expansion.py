import argparse
import json

from llm import LLM
from prompt_config import PromptConfig
from retriever import Retriever


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

    for attempt in range(prompt_conf.retries):
        try:
            generate_text = "{" + llm.generate(body)
            print(generate_text)
            query_expanded = json.loads(generate_text)
            return query_expanded
        except json.JSONDecodeError:
            if attempt < prompt_conf.retries - 1:
                print(
                    f"Failed to decode JSON, retrying... (Attempt {attempt + 1}/{prompt_conf.retries})"
                )
                continue
            else:
                raise Exception("Failed to decode JSON after several retries.")


def main(args: argparse.Namespace) -> None:
    config_llm_path = "../config/llm/claude-3_cofig.yaml"
    # config_llm_path = "../config/llm/command-r-plus_config.yaml"
    config_llm_expansion_path = "../config/llm/claude-3_query_expansion_config.yaml"
    template_path = "../config/prompt_template/prompt_template.yaml"
    template_query_expansion_path = "../config/prompt_template/query_expansion.yaml"
    query_path = "../config/query/query.yaml"

    retriever = Retriever(args.kb_id, args.region)

    prompt_conf = PromptConfig(
        config_llm_expansion_path,
        template_query_expansion_path,
        query_path,
        is_query_expansion=True,
    )
    llm = LLM(args.region, prompt_conf.model_id, prompt_conf.is_stream)

    # Query Expansion
    query_expanded = query_expansion(llm, prompt_conf)
    print(query_expanded)

    # Retrival


if __name__ == "__main__":
    args = get_args()
    main(args)
