import json
import os
from typing import List

import mlflow
import numpy as np
import pandas as pd
from llama_index.core.evaluation import (
    EmbeddingQAFinetuneDataset,
    RetrieverEvaluator,
    generate_question_context_pairs,
)
from llama_index.core.indices.vector_store.retrievers.retriever import (
    VectorIndexRetriever,
)
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI
from loguru import logger

from src.run.cfg import RunConfig


class RetrievalEvaluator:
    def generate_synthetic_dataset(self, cfg: RunConfig, nodes: List[TextNode]):
        retrieval_num_sample_nodes = min(
            len(nodes), cfg.eval_cfg.retrieval_num_sample_nodes
        )
        cfg.eval_cfg.retrieval_num_sample_nodes = retrieval_num_sample_nodes

        if cfg.args.RECREATE_RETRIEVAL_EVAL_DATASET or not os.path.exists(
            cfg.eval_cfg.retrieval_eval_dataset_fp
        ):
            cfg.eval_cfg.retrieval_eval_dataset_fp = (
                f"{cfg.notebook_cache_dp}/retrieval_synthetic_eval_dataset.json"
            )
            logger.info(
                f"Creating new retrieval eval dataset at {cfg.eval_cfg.retrieval_eval_dataset_fp}..."
            )
            if retrieval_num_sample_nodes:
                logger.info(
                    f"Sampling {retrieval_num_sample_nodes} nodes for retrieval evaluation..."
                )
                np.random.seed(41)
                retrieval_eval_nodes = np.random.choice(
                    nodes, retrieval_num_sample_nodes
                )
            else:
                logger.info(f"Using all nodes for retrieval evaluation")
                retrieval_eval_nodes = nodes

            qa_generate_prompt_tmpl = f"""
Context information is below.

---------------------
{{context_str}}
---------------------

{cfg.eval_cfg.retrieval_question_gen_query}
            """

            # Use good model to generate the eval dataset
            retrieval_eval_llm = OpenAI(
                model=cfg.eval_cfg.retrieval_eval_llm_model,
                **cfg.eval_cfg.retrieval_eval_llm_model_config,
            )

            logger.info(f"Creating new synthetic retrieval eval dataset...")
            retrieval_eval_dataset = generate_question_context_pairs(
                retrieval_eval_nodes,
                llm=retrieval_eval_llm,
                num_questions_per_chunk=cfg.eval_cfg.retrieval_num_questions_per_chunk,
                qa_generate_prompt_tmpl=qa_generate_prompt_tmpl,
            )
            logger.info(
                f"Persisting synthetic retrieval eval dataset to {cfg.eval_cfg.retrieval_eval_dataset_fp}..."
            )
            retrieval_eval_dataset.save_json(cfg.eval_cfg.retrieval_eval_dataset_fp)
        else:
            logger.info(
                f"Loading retrieval_eval_nodes from {cfg.eval_cfg.retrieval_eval_dataset_fp}..."
            )
            with open(cfg.eval_cfg.retrieval_eval_dataset_fp, "r") as f:
                retrieval_eval_nodes = json.load(f)

            logger.info(
                f"Loading existing synthetic retrieval eval dataset at {cfg.eval_cfg.retrieval_eval_dataset_fp}..."
            )
            retrieval_eval_dataset = EmbeddingQAFinetuneDataset.from_json(
                cfg.eval_cfg.retrieval_eval_dataset_fp
            )

        self.retrieval_eval_nodes = retrieval_eval_nodes
        self.retrieval_eval_dataset = retrieval_eval_dataset

    async def aevaluate(self, cfg: RunConfig, retriever: VectorIndexRetriever):
        retrieval_metrics = cfg.eval_cfg.retrieval_metrics

        retriever_evaluator = RetrieverEvaluator.from_metric_names(
            retrieval_metrics, retriever=retriever
        )

        retrieval_eval_results = await retriever_evaluator.aevaluate_dataset(
            self.retrieval_eval_dataset
        )

        self.metric_prefix = f"top_{cfg.retrieval_cfg.retrieval_top_k}_retrieval_eval"
        retrieval_eval_results_df, retrieval_eval_results_full_df = (
            self.display_results(
                self.metric_prefix,
                retrieval_eval_results,
                metrics=retrieval_metrics,
            )
        )
        self.retrieval_eval_results_df = retrieval_eval_results_df
        self.retrieval_eval_results_full_df = retrieval_eval_results_full_df

        return retrieval_eval_results_df, retrieval_eval_results_full_df

    def display_results(
        self,
        name,
        eval_results,
        metrics=["hit_rate", "mrr"],
        include_cohere_rerank=False,
    ):
        """Display results from evaluate."""

        eval_dicts = []
        for eval_result in eval_results:
            metric_dict = eval_result.metric_vals_dict
            eval_dict = {
                "query": eval_result.query,
                "expected_ids": eval_result.expected_ids,
                "retrieved_ids": eval_result.retrieved_ids,
                **metric_dict,
            }
            eval_dicts.append(eval_dict)

        full_df = pd.DataFrame(eval_dicts)

        columns = {
            "retrievers": [name],
            **{k: [full_df[k].mean()] for k in metrics},
        }

        if include_cohere_rerank:
            crr_relevancy = full_df["cohere_rerank_relevancy"].mean()
            columns.update({"cohere_rerank_relevancy": [crr_relevancy]})

        metric_df = pd.DataFrame(columns)

        return metric_df, full_df

    def log_to_mlflow(self, cfg: RunConfig):
        notebook_cache_dp = cfg.notebook_cache_dp

        for metric, metric_value in self.retrieval_eval_results_df.to_dict(
            orient="records"
        )[0].items():
            if metric in cfg.eval_cfg.retrieval_metrics:
                mlflow.log_metric(f"{self.metric_prefix}_{metric}", metric_value)
        self.retrieval_eval_results_full_df.to_html(
            f"{notebook_cache_dp}/retrieval_eval_results_full_df.html"
        )
        mlflow.log_artifact(
            f"{notebook_cache_dp}/retrieval_eval_results_full_df.html",
            "retrieval_eval_results_full_df",
        )
