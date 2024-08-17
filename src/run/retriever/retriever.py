from llama_index.core.retrievers import VectorIndexRetriever

from src.run.cfg import RunConfig


def get_retriever(cfg: RunConfig, index):
    vector_retriever = VectorIndexRetriever(
        index=index,
        vector_store_query_mode="mmr",
        similarity_top_k=cfg.retrieval_cfg.retrieval_dense_top_k,
    )
    return vector_retriever
