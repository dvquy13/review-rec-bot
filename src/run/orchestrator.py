import os
from loguru import logger

import qdrant_client
from qdrant_client.models import Distance, VectorParams


from src.run.cfg import RunConfig


class RunOrchestrator:
    @classmethod
    def setup_db(cls, cfg: RunConfig, db: qdrant_client.QdrantClient):
        db_collection = cfg.db_collection
        nodes_persist_fp = cfg.nodes_persist_fp
        recreate_index = cfg.args.RECREATE_INDEX
        embed_model_dim = cfg.llm_cfg.embedding_model_dim

        collection_exists = db.collection_exists(db_collection)
        if recreate_index or not collection_exists:
            if collection_exists:
                logger.info(f"Deleting existing Qdrant collection {db_collection}...")
                db.delete_collection(db_collection)
            if os.path.exists(nodes_persist_fp):
                logger.info(f"Deleting persisted nodes object at {nodes_persist_fp}...")
                os.remove(nodes_persist_fp)
            logger.info(f"Creating new Qdrant collection {db_collection}...")
            db.create_collection(
                db_collection,
                vectors_config=VectorParams(
                    size=embed_model_dim, distance=Distance.COSINE
                ),
            )
        else:
            logger.info(f"Use existing Qdrant collection")
