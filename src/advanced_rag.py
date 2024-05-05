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


def main(args: argparse.Namespace) -> None:
    config_llm_path = "../config/llm/claude-3_cofig.yaml"
    config_llm_expansion_path = "../config/llm/claude-3_query_expansion_config.yaml"
    # config_llm_path = "../config/llm/command-r-plus_config.yaml"
    # config_llm_expansion_path = (
    #     "../config/llm/command-r-plus_query_expansion_config.yaml"
    # )
    template_path = "../config/prompt_template/prompt_template.yaml"
    template_query_expansion_path = "../config/prompt_template/query_expansion.yaml"
    query_path = "../config/query/query.yaml"

    retriever = Retriever(args.kb_id, args.region)

    # step1. Query Expansion
    prompt_conf = PromptConfig(
        config_llm_expansion_path,
        template_query_expansion_path,
        query_path,
        is_query_expansion=True,
    )
    llm = LLM(args.region, prompt_conf.model_id, prompt_conf.is_stream)
    query_expanded = llm.query_expansion(prompt_conf)
    # print(query_expanded)

    # step2. Retrival contexts
    multi_retrieval_results = retriever.retrieve_multiple_queries(
        args.kb_id, args.region, query_expanded
    )
    multi_contexts = retriever.get_multiple_contexts(multi_retrieval_results)
    # print(multi_contexts)
    # print(len(multi_contexts))

    # step3. Augument prompt
    prompt_conf = PromptConfig(config_llm_path, template_path, query_path)
    prompt_conf.format_prompt({"contexts": multi_contexts, "query": prompt_conf.query})
    prompt_conf.format_message({"prompt": prompt_conf.prompt})
    body = json.dumps(prompt_conf.config)

    # step4. Generate message
    generated_text = llm.generate(body)
    print(generated_text)


if __name__ == "__main__":
    args = get_args()
    main(args)
