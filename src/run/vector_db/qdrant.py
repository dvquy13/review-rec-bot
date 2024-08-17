import os
import shutil

import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from loguru import logger
from qdrant_client.models import Distance, VectorParams

from src.run.cfg import RunConfig

from .base import VectorDBWrapper


class QdrantVectorDB(VectorDBWrapper):
    def __init__(self, cfg: RunConfig) -> None:
        self.db_collection_name = cfg.db_collection
        self.storage_context_persist_dp = cfg.storage_context_persist_dp
        self.recreate_index = cfg.args.RECREATE_INDEX
        self.embed_model_dim = cfg.llm_cfg.embedding_model_dim

        self.db_client = qdrant_client.QdrantClient(host="localhost", port=6333)
        self.async_db_client = qdrant_client.AsyncQdrantClient(
            host="localhost", port=6333
        )

        self._setup_db()

        self.vector_store = QdrantVectorStore(
            client=self.db_client,
            collection_name=self.db_collection_name,
            aclient=self.async_db_client,
            enable_hybrid=False,
            prefer_grpc=True,
        )

    def _setup_db(self):
        collection_exists = self.db_client.collection_exists(self.db_collection_name)
        if self.recreate_index or not collection_exists:
            if collection_exists:
                logger.info(
                    f"Deleting existing Qdrant collection {self.db_collection_name}..."
                )
                self.db_client.delete_collection(self.db_collection_name)
            if os.path.exists(self.storage_context_persist_dp):
                logger.info(
                    f"Deleting persisted storage context at {self.storage_context_persist_dp}..."
                )
                shutil.rmtree(self.storage_context_persist_dp)
            logger.info(f"Creating new Qdrant collection {self.db_collection_name}...")
            self.db_client.create_collection(
                self.db_collection_name,
                vectors_config=VectorParams(
                    size=self.embed_model_dim,
                    distance=Distance.COSINE,
                ),
            )
        else:
            logger.info(f"Using existing Qdrant collection: {self.db_collection_name}")

    @property
    def doc_count(self):
        db_collection = self.db_client.get_collection(self.db_collection_name)
        return db_collection.points_count
