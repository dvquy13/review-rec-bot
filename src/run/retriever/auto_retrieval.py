from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo

from src.features.auto_retrieval.prompts import CUSTOM_VECTOR_STORE_QUERY_PROMPT_TMPL
from src.run.cfg import RunConfig

vector_store_info = VectorStoreInfo(
    content_info="Detailed reviews of businesses by users",
    metadata_info=[
        MetadataInfo(
            name="review_id",
            type="str",
            description="Unique identifier for the review.",
        ),
        MetadataInfo(
            name="user_id",
            type="str",
            description="Unique identifier for the user who wrote the review.",
        ),
        MetadataInfo(
            name="business_id",
            type="str",
            description="Unique identifier for the business being reviewed.",
        ),
        MetadataInfo(
            name="review_stars",
            type="int",
            description="Star rating given by the user, ranging from 1 to 5.",
        ),
        MetadataInfo(
            name="useful",
            type="int",
            description="Number of 'useful' votes the review received.",
        ),
        MetadataInfo(
            name="funny",
            type="int",
            description="Number of 'funny' votes the review received.",
        ),
        MetadataInfo(
            name="cool",
            type="int",
            description="Number of 'cool' votes the review received.",
        ),
        MetadataInfo(
            name="date",
            type="str",
            description="Date and time when the review was posted.",
        ),
        MetadataInfo(
            name="biz_name",
            type="str",
            description="Name of the business being reviewed.",
        ),
        MetadataInfo(
            name="biz_address",
            type="str",
            description="Street address of the business.",
        ),
        MetadataInfo(
            name="biz_city",
            type="str",
            description="City where the business is located.",
        ),
        MetadataInfo(
            name="biz_state",
            type="str",
            description="State where the business is located.",
        ),
        MetadataInfo(
            name="biz_postal_code",
            type="str",
            description="Postal code of the business location.",
        ),
        MetadataInfo(
            name="biz_latitude",
            type="float",
            description="Latitude coordinate of the business location.",
        ),
        MetadataInfo(
            name="biz_longitude",
            type="float",
            description="Longitude coordinate of the business location.",
        ),
        MetadataInfo(
            name="biz_stars",
            type="float",
            description="Average star rating of the business.",
        ),
        MetadataInfo(
            name="biz_review_count",
            type="int",
            description="Total number of reviews the business has received.",
        ),
        MetadataInfo(
            name="biz_is_open",
            type="int",
            description="Indicates whether the business is currently open (1) or closed (0).",
        ),
        MetadataInfo(
            name="biz_attributes",
            type="str",
            description="Attributes of the business, such as amenities and services offered.",
        ),
        MetadataInfo(
            name="biz_categories",
            type="str",
            description="Categories the business belongs to, separated by commas.",
        ),
    ],
)


def get_retriever(cfg: RunConfig, index):
    vector_retriever = VectorIndexAutoRetriever(
        index=index,
        vector_store_info=vector_store_info,
        prompt_template_str=CUSTOM_VECTOR_STORE_QUERY_PROMPT_TMPL,
        vector_store_query_mode="mmr",
        similarity_top_k=cfg.retrieval_cfg.retrieval_dense_top_k,
    )
    return vector_retriever
