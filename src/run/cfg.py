import os
import sys
from typing import List, Literal

from loguru import logger
from pydantic import BaseModel

from src.run.args import RunInputArgs
from src.run.utils import pprint_pydantic_model, substitute_punctuation


class LLMConfig(BaseModel):
    llm_provider: Literal["openai", "togetherai", "ollama"] = "togetherai"
    llm_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    embedding_provider: Literal["openai", "togetherai", "ollama", "huggingface"] = (
        "huggingface"
    )
    # embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    embedding_model_name: str = (
        "Snowflake/snowflake-arctic-embed-m-v1.5"  # one-third the size of bge-large-en-v1.5 but better average retrieval benchmark
    )
    embedding_model_dim: int = None

    ollama__host: str = "192.168.100.14"
    ollama__port: int = 11434


class RetrievalConfig(BaseModel):
    retrieval_top_k: int = 10
    retrieval_similarity_cutoff: int = None
    rerank_top_k: int = 5
    rerank_model_name: str = "BAAI/bge-reranker-large"


class EvalConfig(BaseModel):
    retrieval_num_sample_nodes: int = 20
    retrieval_eval_llm_model: str = "gpt-4o-mini"
    retrieval_eval_llm_model_config: dict = {"temperature": 0.3}
    retrieval_num_questions_per_chunk: int = 1
    retrieval_metrics: List[int] = [
        "hit_rate",
        "mrr",
        "precision",
        "recall",
        "ap",
        "ndcg",
    ]
    retrieval_eval_dataset_fp: str = (
        "data/001_retriever/retrieval_synthetic_eval_dataset.json"
    )

    retrieval_question_gen_query: str = """
You are a helpful Retrieval Evaluator.

Your task is to generate {num_questions_per_chunk} search queries based to only the given context, not prior information to assess the accuracy/relevancy of an information retrieval system.
The information retrieval system would then be asked your generated search queries and assessed on how well it can look up and return the correct reviews as context.

<EXAMPLE>
Input context: What a great addition to the Funk Zone!  Grab a bite, grab some tastings, life is good. Right next door to the Santa Barbara Wine Collective, in fact it actually shares the same tables.  We had a fabulous savory croissant.
Output search queries: What are the recommended restaurants nearby Santa Barbara?
</EXAMPLE>

IMPORTANT RULES:
- The generated queries should indicate a clear intention with "finding a restaurant" a typical one
- Restrict the generated queries to the context information provided
- Do not mention anything about the context in the generated queries
- The generated search queries should be specific enough so that there are a few of the relevant reviews only
- Avoid general/ambiguous queries like: "atmosphere and ambience of the place"
"""

    response_question_gen_query: str = """
You are a helpful Retrieval Evaluator.

Your task is to generate {num_questions_per_chunk} search queries based to only the given context, not prior information to assess the accuracy/relevancy of an information retrieval system.
The information retrieval system would then be asked your generated search queries and assessed on how well it can look up and return the correct reviews as context.

<EXAMPLE>
Input context: What a great addition to the Funk Zone!  Grab a bite, grab some tastings, life is good. Right next door to the Santa Barbara Wine Collective, in fact it actually shares the same tables.  We had a fabulous savory croissant.
Output search queries: What are the recommended restaurants nearby Santa Barbara?
</EXAMPLE>

IMPORTANT RULES:
- The generated queries should indicate a clear intention with "finding a restaurant" a typical one
- Restrict the generated queries to the context information provided
- Do not mention anything about the context in the generated queries
- The generated search queries should be specific enough so that there are a few of the relevant reviews only
- Avoid general/ambiguous queries like: "atmosphere and ambience of the place"
"""

    response_synthetic_eval_dataset_fp: str = (
        "data/001_retriever/response_synthetic_eval_dataset.json"
    )
    response_curated_eval_dataset_fp: str = (
        "data/011_analyze_context/response_curated_eval_dataset.json"
    )
    response_eval_llm_model: str = "gpt-4o-mini"
    response_eval_llm_model_config: dict = {"temperature": 0.3}
    response_synthetic_num_questions_per_chunk: int = 1
    response_num_sample_documents: int = 20


class RunConfig(BaseModel):
    args: RunInputArgs = None
    app_name: str = "review_rec_bot"
    db_collection: str = (
        "review_rec_bot__huggingface__Snowflake_snowflake_arctic_embed_m_v1_5__005_use_smaller_embedding_model"
    )
    nodes_persist_fp: str = "data/005_use_smaller_embedding_model/nodes.pkl"
    notebook_cache_dp: str = None

    data_fp: str = "../data/yelp_dataset/sample/sample_100_biz/denom_review.parquet"

    llm_cfg: LLMConfig = LLMConfig()

    retrieval_cfg: RetrievalConfig = RetrievalConfig()

    eval_cfg: EvalConfig = EvalConfig()

    batch_size: int = 16

    def init(self, args: RunInputArgs):
        self.args = args

        if args.OBSERVABILITY:
            logger.info(f"Starting Observability server with Phoenix...")
            import phoenix as px

            px.launch_app()
            import llama_index.core

            llama_index.core.set_global_handler("arize_phoenix")

        if args.DEBUG:
            logger.info(f"Enabling LlamaIndex DEBUG logging...")
            import logging

            logging.getLogger("llama_index").addHandler(
                logging.StreamHandler(stream=sys.stdout)
            )
            logging.getLogger("llama_index").setLevel(logging.DEBUG)

        if args.LOG_TO_MLFLOW:
            logger.info(
                f"Setting up MLflow experiment {args.EXPERIMENT_NAME} - run {args.RUN_NAME}..."
            )
            import mlflow

            mlflow.set_experiment(args.EXPERIMENT_NAME)
            mlflow.start_run(run_name=args.RUN_NAME, description=args.RUN_DESCRIPTION)

        self.notebook_cache_dp = f"data/{args.RUN_NAME}"
        logger.info(
            f"Notebook-generated artifacts are persisted at {self.notebook_cache_dp}"
        )
        os.makedirs(self.notebook_cache_dp, exist_ok=True)

        if args.RECREATE_INDEX:
            logger.info(
                f"ARGS.RECREATE_INDEX=True -> Overwriting db_collection and nodes_persist_fp..."
            )
            collection_raw_name = f"{self.app_name}__{self.llm_cfg.embedding_provider}__{self.llm_cfg.embedding_model_name}__{args.RUN_NAME}"
            self.db_collection = substitute_punctuation(collection_raw_name)
            self.nodes_persist_fp = f"{self.notebook_cache_dp}/nodes.pkl"

    def setup_llm(self):
        # Set up LLM
        llm_provider = self.llm_cfg.llm_provider
        llm_model_name = self.llm_cfg.llm_model_name

        if llm_provider == "ollama":
            import subprocess

            from llama_index.llms.ollama import Ollama

            ollama_host = self.llm_cfg.ollama__host
            ollama_port = self.llm_cfg.ollama__port

            base_url = f"http://{ollama_host}:{ollama_port}"
            llm = Ollama(
                base_url=base_url,
                model=llm_model_name,
                request_timeout=60.0,
            )
            command = ["ping", "-c", "1", ollama_host]
            subprocess.run(command, capture_output=True, text=True)
        elif llm_provider == "openai":
            from llama_index.llms.openai import OpenAI

            llm = OpenAI(model=llm_model_name)
        elif llm_provider == "togetherai":
            from llama_index.llms.together import TogetherLLM

            llm = TogetherLLM(model=llm_model_name)

        # Set up Embedding Model
        embedding_provider = self.llm_cfg.embedding_provider
        embedding_model_name = self.llm_cfg.embedding_model_name

        if embedding_provider == "huggingface":
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        elif embedding_provider == "openai":
            from llama_index.embeddings.openai import OpenAIEmbedding

            embed_model = OpenAIEmbedding()
        elif embedding_provider == "togetherai":
            from llama_index.embeddings.together import TogetherEmbedding

            embed_model = TogetherEmbedding(embedding_model_name)
        elif embedding_provider == "ollama":
            from llama_index.embeddings.ollama import OllamaEmbedding

            embed_model = OllamaEmbedding(
                model_name=embedding_model_name,
                base_url=base_url,
                ollama_additional_kwargs={"mirostat": 0},
            )

        self.llm_cfg.embedding_model_dim = len(
            embed_model.get_text_embedding("sample text")
        )

        return llm, embed_model

    def __repr__(self):
        return pprint_pydantic_model(self)

    def __str__(self):
        return pprint_pydantic_model(self)
