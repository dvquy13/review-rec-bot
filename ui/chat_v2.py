import os
import sys
import time

import chainlit as cl
import qdrant_client
import Stemmer
import torch
from dotenv import load_dotenv
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import CallbackManager
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from loguru import logger

sys.path.insert(0, "..")


from src.features.append_reference.custom_query_engine import (
    ManualAppendReferenceQueryEngine,
)
from src.features.synthesize_recommendation.custom_tree_summarize import (
    CUSTOM_TREE_SUMMARIZE_PROMPT_SEL,
)
from src.run.args import RunInputArgs
from src.run.cfg import RunConfig
from src.run.orchestrator import RunOrchestrator

load_dotenv()

USE_GPU = True

if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

logger.info(f"{torch.cuda.is_available()=}")

ARGS = RunInputArgs(
    EXPERIMENT_NAME="Review Rec Bot - Yelp Review Rec Bot",
    RUN_NAME="026_chatbot_interface",
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
cfg.storage_context_persist_dp = (
    f"{dir_prefix}/data/018_finetuned_embedding_reindex/storage_context"
)
cfg.db_collection = "review_rec_bot__018_finetuned_embedding_reindex__huggingface____data_finetune_embedding_finetuned_model"
cfg.llm_cfg.embedding_model_name = (
    "../notebooks/data/finetune_embedding/finetuned_model"
)

cfg.init(ARGS)

logger.info(cfg)

llm, embed_model = cfg.setup_llm()

logger.info(cfg.llm_cfg.model_dump_json(indent=2))

Settings.embed_model = embed_model
Settings.llm = llm

qdrantdb = qdrant_client.QdrantClient(host="localhost", port=6333)
aqdrantdb = qdrant_client.AsyncQdrantClient(host="localhost", port=6333)

RunOrchestrator.setup_db(cfg, qdrantdb)

db_collection = qdrantdb.get_collection(cfg.db_collection)
vector_store = QdrantVectorStore(
    client=qdrantdb,
    collection_name=cfg.db_collection,
    aclient=aqdrantdb,
    enable_hybrid=False,
    prefer_grpc=True,
)


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
    # sparse_top_k=cfg.retrieval_cfg.retrieval_sparse_top_k,
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

response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.TREE_SUMMARIZE,
    summary_template=CUSTOM_TREE_SUMMARIZE_PROMPT_SEL,
)
query_engine = ManualAppendReferenceQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=node_postprocessors,
)

logger.info(f"Registerring Query Engine as Tool...")
query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="reco_review",
        description=(
            "useful for when you want to find places to visit"
            " based on end-user reviews. Takes input in a question"
            " format, e.g.: What are the best Vietnamese restaurants in Texas?"
        ),
    ),
)

tools = [query_engine_tool]

agent_system_prompt = """
You're a helpful assistant who excels at recommending places to go.

Always return the referenced paragraphs at the end of your answer to users. Format them nicely if need to.
"""


@cl.on_chat_start
async def start():
    agent = OpenAIAgent.from_tools(
        tools,
        verbose=True,
        system_prompt=agent_system_prompt,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )

    cl.user_session.set("agent", agent)

    await cl.Message(
        author="Jaina",
        content="Hello! I'm Jaina. I help people find places to visit. What are you looking for today?",
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
