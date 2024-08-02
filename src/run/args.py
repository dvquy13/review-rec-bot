from pydantic import BaseModel

from src.run.utils import pprint_pydantic_model


class RunInputArgs(BaseModel):
    EXPERIMENT_NAME: str
    RUN_NAME: str
    RUN_DESCRIPTION: str

    TESTING: bool = False
    DEBUG: bool = False
    OBSERVABILITY: bool = True
    LOG_TO_MLFLOW: bool = False

    RECREATE_INDEX: bool = False
    RECREATE_RETRIEVAL_EVAL_DATASET: bool = False
    RECREATE_RESPONSE_EVAL_DATASET: bool = False

    def __repr__(self):
        return pprint_pydantic_model(self)

    def __str__(self):
        return pprint_pydantic_model(self)
