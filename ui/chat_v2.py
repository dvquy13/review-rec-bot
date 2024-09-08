import os
import sys

import chainlit as cl
import pandas as pd
import Stemmer
import torch
from dotenv import load_dotenv
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.retrievers.bm25 import BM25Retriever
from loguru import logger

sys.path.insert(0, "..")

from src.features.citation.custom_citation_query_engine import (
    CUSTOM_CITATION_QA_TEMPLATE,
    CUSTOM_CITATION_REFINE_TEMPLATE,
)
from src.run.args import RunInputArgs
from src.run.cfg import RunConfig
from src.svc.availability.availability_check import ReservationService
from src.svc.current_datetime.current_datetime_check import get_current_datetime
from ui.callback_handler import LlamaIndexCallbackHandler

load_dotenv()

USE_GPU = True

if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

logger.info(f"{torch.cuda.is_available()=}")

ARGS = RunInputArgs(
    EXPERIMENT_NAME="Review Rec Bot - Yelp Review Rec Bot",
    RUN_NAME="034_rerun_400_restaurants",
    RUN_DESCRIPTION="""
# Objective

# Implementation

# Changelog
""",
    TESTING=False,
    LOG_TO_MLFLOW=False,
    OBSERVABILITY=True,
    RECREATE_INDEX=False,
    RECREATE_RETRIEVAL_EVAL_DATASET=False,
    RECREATE_RESPONSE_EVAL_DATASET=False,
    DEBUG=False,
)

logger.info(ARGS)

cfg = RunConfig()

dir_prefix = "../notebooks"
cfg.storage_context_persist_dp = os.path.abspath(
    f"{dir_prefix}/data/034_rerun_400_restaurants/storage_context"
)
cfg.db_collection = "review_rec_bot__034_rerun_400_restaurants"
cfg.db_collection_fp = "data/034_rerun_400_restaurants/chroma_db"
cfg.llm_cfg.embedding_model_name = os.path.abspath(
    f"{dir_prefix}/data/finetune_embedding/finetuned_model"
)
cfg.data_fp = "../data/yelp_dataset/sample/sample_400_biz/denom_review.parquet"

cfg.init(ARGS)

logger.info(cfg)

llm, embed_model = cfg.setup_llm()

logger.info(cfg.llm_cfg.model_dump_json(indent=2))

Settings.embed_model = embed_model
Settings.llm = llm

if cfg.vector_db == "chromadb":
    from src.run.vector_db import ChromaVectorDB as VectorDB
elif cfg.vector_db == "qdrant":
    from src.run.vector_db import QdrantVectorDB as VectorDB

vector_db = VectorDB(cfg)
vector_store = vector_db.vector_store
db_collection_count = vector_db.doc_count
logger.info(f"{db_collection_count=}")

logger.info(f"Loading Storage Context from {cfg.storage_context_persist_dp}...")
docstore = SimpleDocumentStore.from_persist_dir(
    persist_dir=cfg.storage_context_persist_dp
)
storage_context = StorageContext.from_defaults(
    docstore=docstore, vector_store=vector_store
)
nodes = list(docstore.docs.values())

logger.info(f"[COLLECT] {len(nodes)=}")

logger.info(f"Configuring Vector Retriever...")
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context
)
vector_retriever = VectorIndexRetriever(
    index=index,
    vector_store_query_mode="mmr",
    similarity_top_k=cfg.retrieval_cfg.retrieval_dense_top_k,
)

logger.info(f"Configuring BM25 Retriever...")
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=cfg.retrieval_cfg.retrieval_sparse_top_k,
    stemmer=Stemmer.Stemmer("english"),
    language="english",
)

logger.info(f"Configuring Query Fusion Retriever...")
query_gen_prompt = """
You are a helpful assistant that expands an input query into new strings that aim to increase the recall of an information retrieval system. The strings can be queries or paragraphs or sentences.
You should apply different techniques to create new strings. Here are some example techniques:
- Technique 1 - Optimize for full-text search: Rephrase the input query to contain only important keywords. Remove all stopwords and low information words. Example input query: "What are some places to enjoy cold brew coffee in Hanoi?" -> Expected output:  "cold brew coffee hanoi"
- Technique 2 - Optimize for similarity-based vector retrieval: Create a fake user review that should contain the answer for the question. Example input query: "What are some good Pho restaurants in Singapore?" -> Expected output query: "I found So Pho offerring a variety of choices to enjoy not Pho but some other Vietnamese dishes like bun cha. The price is reasonable."

Generate at least {num_queries} new strings by iterating over the technique in order. For example, your first generated string should always use technique 1, second technique 2. If run of of techniques then re-iterate from the start.

Return one string on each line, related to the input query.

Only return the strings. Never include the chosen technique.

Input Query: {query}\n
New strings:\n
"""

llm = OpenAI(
    model=cfg.eval_cfg.response_eval_llm_model,
    **cfg.eval_cfg.response_eval_llm_model_config,
)

retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    llm=llm,
    similarity_top_k=cfg.retrieval_cfg.retrieval_top_k,
    num_queries=2,  # set this to 1 to disable query generation
    mode="reciprocal_rerank",
    use_async=True,
    verbose=True,
    query_gen_prompt=query_gen_prompt,
)

logger.info(f"Setting up Post-Retriever Processor...")
node_postprocessors = []

if cfg.retrieval_cfg.retrieval_similarity_cutoff is not None:
    node_postprocessors.append(
        SimilarityPostprocessor(
            similarity_cutoff=cfg.retrieval_cfg.retrieval_similarity_cutoff
        )
    )

reranker = FlagEmbeddingReranker(
    model=cfg.retrieval_cfg.rerank_model_name,
    top_n=cfg.retrieval_cfg.rerank_top_k,
    use_fp16=True,
)
node_postprocessors.append(reranker)

query_engine = CitationQueryEngine.from_args(
    index=index,
    retriever=retriever,
    node_postprocessors=node_postprocessors,
    citation_qa_template=CUSTOM_CITATION_QA_TEMPLATE,
    citation_refine_template=CUSTOM_CITATION_REFINE_TEMPLATE,
)

logger.info(f"Registering Query Engine as Tool...")
query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="reco_review",
        description=(
            "useful for when you want to find restaurants and cafes"
            " based on end-user reviews. Takes input in a question"
            " format, e.g.: What are the best Vietnamese restaurants in Texas?"
        ),
    ),
)

logger.info(f"Loading the data file to build mappers for ReservationService...")

data = pd.read_parquet(cfg.data_fp)
data = data.assign(
    biz_categories=lambda df: df["biz_categories"].str.split(", "),
    date=lambda df: df["date"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
)
logger.info(f"{len(data)=}")

opening_hours_db = data.set_index("business_id")["biz_hours"].dropna().to_dict()
biz_name_id_mapper = data.set_index("biz_name")["business_id"].dropna().to_dict()
rez_tool = ReservationService(opening_hours_db, biz_name_id_mapper)

get_current_datetime_tool = FunctionTool.from_defaults(fn=get_current_datetime)

tools = [query_engine_tool] + rez_tool.to_tool_list() + [get_current_datetime_tool]

agent_system_prompt = """
You're a helpful assistant who excels at recommending places to go.

When users ask for relative time like today or tomorrow, always use the get_current_datetime tool.

You should always narrow down the places like states or cities in the US. If you don't know this information, please ask the user.

You must return the cited sources to your users so that they know you base on which information to make the recommendations.

If there are citation sources returned from the tools, always return them exactly as they are of your answer to users.
This mean that you must respect if where the citation numbers (like [1], [2]) in the answers and at the end below the Sources section.
"""


@cl.on_chat_start
async def start():
    agent = OpenAIAgent.from_tools(
        tools,
        verbose=True,
        system_prompt=agent_system_prompt,
        callback_manager=CallbackManager([LlamaIndexCallbackHandler()]),
    )

    cl.user_session.set("agent", agent)

    await cl.Message(
        author="Jaina",
        content="Hello! I'm Jaina. I help people find restaurants and cafes. What are you looking for today?",
    ).send()


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")

    msg = cl.Message(content="", author="Jaina")

    # res = await agent.astream_chat(message.content)  # will not work, asyncio error
    res = await cl.make_async(agent.stream_chat)(message.content)

    for token in res.response_gen:
        await msg.stream_token(token)
    await msg.send()
