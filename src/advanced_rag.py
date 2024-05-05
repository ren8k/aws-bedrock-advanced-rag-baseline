import argparse
import json

from llm import LLM
from prompt_config import PromptConfig
from retriever import Retriever
from utils import load_yaml


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
    parser.add_argument(
        "--config-path",
        type=str,
        default="../config/config.yaml",
        help="The path to the configuration YAML file.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    config = load_yaml(args.config_path)

    retriever = Retriever(args.kb_id, args.region)

    # step1. Query Expansion
    prompt_conf = PromptConfig(
        config["config_llm_expansion_path"],
        config["template_query_expansion_path"],
        config["query_path"],
        is_query_expansion=True,
    )
    llm = LLM(args.region, prompt_conf.model_id, prompt_conf.is_stream)
    queries_expanded = llm.expand_queries(prompt_conf)

    # step2. Retrival contexts
    multi_retrieved_results = retriever.retrieve_parallel(
        args.kb_id, args.region, queries_expanded
    )
    multi_contexts = retriever.get_multiple_contexts(multi_retrieved_results)

    # step3. Relevance evaluation
    prompt_conf = PromptConfig(
        config["config_llm_relevance_eval_path"],
        config["template_relevance_eval_path"],
        config["query_path"],
        is_relevance_eval=True,
    )
    prompts_and_contexts = prompt_conf.create_prompts_for_relevance_eval(
        queries_expanded, multi_contexts
    )
    print(len(prompts_and_contexts))
    multi_contexts = llm.eval_relevance_parallel(
        args.region, prompt_conf.model_id, prompt_conf, prompts_and_contexts
    )
    print(len(multi_contexts))

    # step4. Augument prompt
    prompt_conf = PromptConfig(
        config["config_llm_rag_path"], config["template_rag_path"], config["query_path"]
    )
    prompt_conf.format_prompt({"contexts": multi_contexts, "query": prompt_conf.query})
    prompt_conf.format_message({"prompt": prompt_conf.prompt})
    body = json.dumps(prompt_conf.config)

    # step5. Generate message
    generated_text = llm.generate(body)
    print(generated_text)


if __name__ == "__main__":
    args = get_args()
    main(args)
