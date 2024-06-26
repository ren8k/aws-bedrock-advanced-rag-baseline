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
    parser.add_argument(
        "--relevance-eval",
        action="store_true",
        help="Whether to evaluate the relevance of the contexts.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    config = load_yaml(args.config_path)
    retriever = Retriever(args.kb_id, args.region)

    # step1. Expand query
    prompt_conf = PromptConfig(
        config["template_query_expansion_path"],
        config["query_path"],
        is_query_expansion=True,
    )
    llm_conf = LLMConfig(config["config_llm_expansion_path"])
    llm = LLM(args.region, llm_conf.model_id, llm_conf.is_stream)
    queries_expanded = llm.expand_queries(llm_conf, prompt_conf)
    logger.info(f"Expanded queries: {queries_expanded}")

    # step2. Retrival contexts
    multi_retrieved_results = retriever.retrieve_parallel(
        args.kb_id, args.region, queries_expanded, max_workers=5, no_of_results=5
    )
    multi_contexts = retriever.get_multiple_contexts(multi_retrieved_results)
    logger.info(f"Number of retrieved contexts: {len(multi_contexts)}")

    if args.relevance_eval:
        # step3. Eval relevance
        prompt_conf = PromptConfig(
            config["template_relevance_eval_path"],
            config["query_path"],
            is_relevance_eval=True,
        )
        llm_conf = LLMConfig(config["config_llm_relevance_eval_path"])
        llm = LLM(args.region, llm_conf.model_id, llm_conf.is_stream)
        prompts_and_contexts = prompt_conf.create_prompts_for_relevance_eval(
            multi_contexts
        )
        multi_contexts = llm.eval_relevance_parallel(
            args.region,
            llm_conf,
            prompts_and_contexts,
            max_workers=10,
        )
        logger.info(f"Number of relevant contexts: {len(multi_contexts)}")

    # step4. Augument prompt
    prompt_conf = PromptConfig(config["template_rag_path"], config["query_path"])
    llm_conf = LLMConfig(config["config_llm_rag_path"])
    llm = LLM(args.region, llm_conf.model_id, llm_conf.is_stream)
    prompt_conf.format_prompt({"contexts": multi_contexts, "query": prompt_conf.query})
    llm_conf.format_message(prompt_conf.prompt)
    body = json.dumps(llm_conf.llm_args)

    # step5. Generate message
    generated_text = llm.generate(body)
    logger.info(generated_text)


if __name__ == "__main__":
    args = get_args()
    main(args)
