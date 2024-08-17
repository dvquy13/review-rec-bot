import os
import shutil

import chromadb
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from loguru import logger

from src.run.cfg import RunConfig

from .base import VectorDBWrapper


class ChromaVectorDB(VectorDBWrapper):
    def __init__(self, cfg: RunConfig):
        db_collection_name = cfg.db_collection
        storage_context_persist_dp = cfg.storage_context_persist_dp
        recreate_index = cfg.args.RECREATE_INDEX

        db_client = chromadb.PersistentClient(cfg.db_collection_fp)

        try:
            db_client.get_collection(db_collection_name)
            collection_exists = True
        except ValueError:
            collection_exists = False

        if recreate_index or not collection_exists:
            if collection_exists:
                logger.info(
                    f"Deleting existing ChromaDB collection {db_collection_name}..."
                )
                db_client.delete_collection(db_collection_name)
            if os.path.exists(storage_context_persist_dp):
                logger.info(
                    f"Deleting persisted storage context at {storage_context_persist_dp}..."
                )
                shutil.rmtree(storage_context_persist_dp)
            logger.info(f"Creating new ChromaDB collection {db_collection_name}...")
            db_client.create_collection(
                name=db_collection_name,
            )
        else:
            logger.info(f"Using existing ChromaDB collection: {db_collection_name}")

        self.db_collection = db_client.get_collection(db_collection_name)
        self.vector_store: BasePydanticVectorStore = ChromaVectorStore(
            chroma_collection=self.db_collection
        )

    @property
    def doc_count(self) -> int:
        return self.db_collection.count()
