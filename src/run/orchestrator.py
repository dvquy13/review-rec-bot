import os
import shutil

import qdrant_client
from loguru import logger
from qdrant_client.models import Distance, VectorParams

from src.run.cfg import RunConfig


class RunOrchestrator:
    @classmethod
    def setup_db(cls, cfg: RunConfig, db: qdrant_client.QdrantClient):
        db_collection = cfg.db_collection
        storage_context_persist_dp = cfg.storage_context_persist_dp
        recreate_index = cfg.args.RECREATE_INDEX
        embed_model_dim = cfg.llm_cfg.embedding_model_dim

        collection_exists = db.collection_exists(db_collection)
        if recreate_index or not collection_exists:
            if collection_exists:
                logger.info(f"Deleting existing Qdrant collection {db_collection}...")
                db.delete_collection(db_collection)
            if os.path.exists(storage_context_persist_dp):
                logger.info(
                    f"Deleting persisted storage context at {storage_context_persist_dp}..."
                )
                shutil.rmtree(storage_context_persist_dp)
            logger.info(f"Creating new Qdrant collection {db_collection}...")
            db.create_collection(
                db_collection,
                vectors_config=VectorParams(
                    size=embed_model_dim, distance=Distance.COSINE
                ),
            )
        else:
            logger.info(f"Use existing Qdrant collection: {db_collection}")
