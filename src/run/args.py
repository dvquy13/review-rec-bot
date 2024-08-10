from loguru import logger
from pydantic import BaseModel, ConfigDict, field_validator

from src.run.utils import pprint_pydantic_model


class RunInputArgs(BaseModel):
    EXPERIMENT_NAME: str
    RUN_NAME: str
    RUN_DESCRIPTION: str

    TESTING: bool = False
    DEBUG: bool = False
    OBSERVABILITY: bool = True
    LOG_TO_MLFLOW: bool = False

    RECREATE_RETRIEVAL_EVAL_DATASET: bool = False
    RECREATE_RESPONSE_EVAL_DATASET: bool = False
    # Need to put the RECREATE_INDEX after the other RECREATE is to be able to access the information from those variables and modify them
    RECREATE_INDEX: bool = False

    def __repr__(self):
        return pprint_pydantic_model(self)

    def __str__(self):
        return pprint_pydantic_model(self)

    @field_validator("RECREATE_INDEX")
    def check_recreate_flags(cls, value, values, **kwargs):
        data = values.data
        if value:
            if not data["RECREATE_RETRIEVAL_EVAL_DATASET"]:
                logger.warning(
                    "If RECREATE_INDEX is True, then RECREATE_RETRIEVAL_EVAL_DATASET should also be True. Automatically setting RECREATE_RETRIEVAL_EVAL_DATASET to True."
                )
                data["RECREATE_RETRIEVAL_EVAL_DATASET"] = True
            if not data["RECREATE_RESPONSE_EVAL_DATASET"]:
                logger.warning(
                    "If RECREATE_INDEX is True, then RECREATE_RESPONSE_EVAL_DATASET should also be True. Automatically setting RECREATE_RESPONSE_EVAL_DATASET to True."
                )
                data["RECREATE_RESPONSE_EVAL_DATASET"] = True
        return value
