import json
import os
from typing import List

import mlflow
import numpy as np
import pandas as pd
from llama_index.core.evaluation import (
    ContextRelevancyEvaluator,
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)
from llama_index.core.evaluation.notebook_utils import get_eval_results_df
from llama_index.core.llama_dataset import (
    CreatedBy,
    CreatedByType,
    LabeledRagDataset,
    LabelledRagDataExample,
    LabelledRagDataset,
)
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.schema import Document
from llama_index.llms.openai import OpenAI
from loguru import logger
from tqdm import tqdm

from src.run.cfg import RunConfig
from src.run.eval.manual_eval_dataset import MANUAL_EVAL_QA


class ResponseEvaluator:
    def generate_synthetic_dataset(self, cfg: RunConfig, documents: List[Document]):
        notebook_cache_dp = cfg.notebook_cache_dp
        response_num_sample_documents = min(
            len(documents), cfg.eval_cfg.response_num_sample_documents
        )

        if response_num_sample_documents:
            logger.info(
                f"Sampling {response_num_sample_documents} documents for response evaluation..."
            )
            np.random.seed(41)
            response_eval_documents = np.random.choice(
                documents, response_num_sample_documents
            )
        else:
            logger.info(f"Using all documents for retrieval evaluation")
            response_eval_documents = documents

        if cfg.args.RECREATE_RESPONSE_EVAL_DATASET or not os.path.exists(
            cfg.eval_cfg.response_synthetic_eval_dataset_fp
        ):
            response_synthetic_eval_dataset_fp = (
                f"{notebook_cache_dp}/response_synthetic_eval_dataset.json"
            )
            cfg.eval_cfg.response_synthetic_eval_dataset_fp = (
                response_synthetic_eval_dataset_fp
            )
            logger.info(
                f"Creating new response eval dataset at {response_synthetic_eval_dataset_fp}..."
            )
            logger.info(f"Creating synthetic response eval dataset...")
            # Use good model to generate the eval dataset
            from llama_index.llms.openai import OpenAI

            response_eval_llm = OpenAI(
                model=cfg.eval_cfg.response_eval_llm_model,
                **cfg.eval_cfg.response_eval_llm_model_config,
            )

            # instantiate a DatasetGenerator
            response_dataset_generator = RagDatasetGenerator.from_documents(
                response_eval_documents,
                llm=response_eval_llm,
                num_questions_per_chunk=cfg.eval_cfg.response_synthetic_num_questions_per_chunk,  # set the number of questions per nodes
                question_gen_query=cfg.eval_cfg.response_question_gen_query,  # Reuse the same format from the above Retrieval Question Gen Query
                show_progress=True,
                workers=(os.cpu_count() - 1),
            )

            response_synthetic_eval_dataset = (
                response_dataset_generator.generate_dataset_from_nodes()
            )

            logger.info(
                f"Persisting synthetic response eval dataset at {response_synthetic_eval_dataset_fp}..."
            )
            response_synthetic_eval_dataset.save_json(
                response_synthetic_eval_dataset_fp
            )
        else:
            response_synthetic_eval_dataset_fp = (
                cfg.eval_cfg.response_synthetic_eval_dataset_fp
            )
            logger.info(
                f"Loading existing synthetic response eval dataset at {response_synthetic_eval_dataset_fp}..."
            )
            response_synthetic_eval_dataset = LabeledRagDataset.from_json(
                response_synthetic_eval_dataset_fp
            )

        return response_eval_documents, response_synthetic_eval_dataset

    def generate_curated_dataset(self, cfg: RunConfig):
        examples = []

        for question, expected_anwser in MANUAL_EVAL_QA:
            example = LabelledRagDataExample(
                query=question,
                query_by=CreatedBy(type=CreatedByType.HUMAN),
                reference_answer=expected_anwser,
                reference_answer_by=CreatedBy(type=CreatedByType.HUMAN),
                reference_contexts=[],
            )
            examples.append(example)

        response_curated_eval_dataset = LabelledRagDataset(examples=examples)

        # save this dataset as it is required for the submission
        response_curated_eval_dataset_fp = (
            f"{cfg.notebook_cache_dp}/response_curated_eval_dataset.json"
        )
        cfg.eval_cfg.response_curated_eval_dataset_fp = response_curated_eval_dataset_fp
        logger.info(
            f"Persisting curated response eval dataset at {response_curated_eval_dataset_fp}..."
        )
        response_curated_eval_dataset.save_json(response_curated_eval_dataset_fp)

        return response_curated_eval_dataset

    def evaluate_labelled_rag_dataset(
        self,
        response_eval_dataset,
        response_eval_prediction_dataset,
        dataset_name="synthetic",
        judge_model="gpt-4o-mini",
        cache_dp=".",
    ):
        # Instantiate the judges
        judges = {
            "correctness": CorrectnessEvaluator(
                llm=OpenAI(temperature=0, model=judge_model),
            ),
            "relevancy": RelevancyEvaluator(
                llm=OpenAI(temperature=0, model=judge_model),
            ),
            "faithfulness": FaithfulnessEvaluator(
                llm=OpenAI(temperature=0, model=judge_model),
            ),
            "context_relevancy": ContextRelevancyEvaluator(
                llm=OpenAI(temperature=0, model=judge_model),
            ),
        }

        # Initialize evaluations dictionary
        evals = {
            "correctness": [],
            "relevancy": [],
            "faithfulness": [],
            "context_relevancy": [],
            "contexts": [],
        }

        # Evaluate each prediction
        for example, prediction in tqdm(
            zip(
                response_eval_dataset.examples,
                response_eval_prediction_dataset.predictions,
            ),
            total=len(response_eval_dataset.examples),
        ):
            correctness_result = judges["correctness"].evaluate(
                query=example.query,
                response=prediction.response,
                reference=example.reference_answer,
            )

            relevancy_result = judges["relevancy"].evaluate(
                query=example.query,
                response=prediction.response,
                contexts=prediction.contexts,
            )

            faithfulness_result = judges["faithfulness"].evaluate(
                query=example.query,
                response=prediction.response,
                contexts=prediction.contexts,
            )

            context_relevancy_result = judges["context_relevancy"].evaluate(
                query=example.query,
                response=prediction.response,
                contexts=prediction.contexts,
            )

            evals["correctness"].append(correctness_result)
            evals["relevancy"].append(relevancy_result)
            evals["faithfulness"].append(faithfulness_result)
            evals["context_relevancy"].append(context_relevancy_result)
            evals["contexts"].append(prediction.contexts)

        # Save evaluations to JSON
        evaluations_objects = {
            "correctness": [e.dict() for e in evals["correctness"]],
            "faithfulness": [e.dict() for e in evals["faithfulness"]],
            "relevancy": [e.dict() for e in evals["relevancy"]],
            "contexts": evals["contexts"],
        }

        with open(f"{cache_dp}/{dataset_name}_evaluations.json", "w") as json_file:
            json.dump(evaluations_objects, json_file)

        # Generate evaluation results DataFrames
        deep_eval_correctness_df, mean_correctness_df = get_eval_results_df(
            ["base_rag"] * len(evals["correctness"]),
            evals["correctness"],
            metric="correctness",
        )
        deep_eval_relevancy_df, mean_relevancy_df = get_eval_results_df(
            ["base_rag"] * len(evals["relevancy"]),
            evals["relevancy"],
            metric="relevancy",
        )
        deep_eval_faithfulness_df, mean_faithfulness_df = get_eval_results_df(
            ["base_rag"] * len(evals["faithfulness"]),
            evals["faithfulness"],
            metric="faithfulness",
        )
        deep_eval_context_relevancy_df, mean_context_relevancy_df = get_eval_results_df(
            ["base_rag"] * len(evals["context_relevancy"]),
            evals["context_relevancy"],
            metric="context_relevancy",
        )

        mean_scores_df = pd.concat(
            [
                mean_correctness_df.reset_index(),
                mean_relevancy_df.reset_index(),
                mean_faithfulness_df.reset_index(),
                mean_context_relevancy_df.reset_index(),
            ],
            axis=0,
            ignore_index=True,
        )
        mean_scores_df = mean_scores_df.set_index("index")
        mean_scores_df.index = mean_scores_df.index.set_names(["metrics"])

        deep_eval_df = pd.concat(
            [
                deep_eval_correctness_df[["query", "answer"]],
                deep_eval_relevancy_df[["scores"]].rename(
                    columns={"scores": "relevancy_score"}
                ),
                deep_eval_correctness_df[["scores"]].rename(
                    columns={"scores": "correctness_score"}
                ),
                deep_eval_faithfulness_df[["scores"]].rename(
                    columns={"scores": "faithfulness_score"}
                ),
                deep_eval_context_relevancy_df[["scores"]].rename(
                    columns={"scores": "context_relevancy_score"}
                ),
                pd.Series(evals["contexts"], name="contexts"),
            ],
            axis=1,
        )

        return mean_scores_df, deep_eval_df

    def log_to_mlflow(
        self, cfg: RunConfig, dataset_name: str, mean_scores_df, deep_eval_df
    ):
        notebook_cache_dp = cfg.notebook_cache_dp

        for k, v in mean_scores_df.T.to_dict(orient="records")[0].items():
            mlflow.log_metric(f"response_{dataset_name}_eval__{k}", v)
        deep_eval_df.to_html(f"{notebook_cache_dp}/{dataset_name}_deep_eval_df.html")
        mlflow.log_artifact(
            f"{notebook_cache_dp}/{dataset_name}_deep_eval_df.html",
            f"{dataset_name}_deep_eval_df",
        )
