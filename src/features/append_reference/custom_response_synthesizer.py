from typing import Any, Sequence
from loguru import logger
from pprint import pformat

from llama_index.core.prompts.base import PromptTemplate, SelectorPromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.prompts.default_prompt_selectors import (
    default_text_qa_conditionals,
)
from llama_index.core.response_synthesizers.compact_and_refine import CompactAndRefine
from llama_index.core.types import RESPONSE_TEXT_TYPE


# # text qa prompt
# TEXT_QA_SYSTEM_PROMPT = ChatMessage(
#     content=(
#         "You are an expert Q&A system that is trusted around the world.\n"
#         "Always answer the query using the provided context information, "
#         "and not prior knowledge.\n"
#         "Some rules to follow:\n"
#         "1. Never directly reference the given context in your answer.\n"
#         "2. Avoid statements like 'Based on the context, ...' or "
#         "'The context information ...' or anything along "
#         "those lines."
#     ),
#     role=MessageRole.SYSTEM,
# )

# TEXT_QA_PROMPT_TMPL_MSGS = [
#     TEXT_QA_SYSTEM_PROMPT,
#     ChatMessage(
#         content=(
#             "Context information is below.\n"
#             "---------------------\n"
#             "{context_str}\n"
#             "---------------------\n"
#             "Given the context information and not prior knowledge, "
#             "answer the query.\n"
#             "Query: {query_str}\n"
#             "Answer: "
#         ),
#         role=MessageRole.USER,
#     ),
# ]

# CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

TEXT_QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Compile a list of referenced title and url copying exactly from the corresponding fields in the given context "
    "and format nicely into a section named `Sources` appended to the end of your answer.\n"
    "Never propose or make up new references.\n"
    "Query: {query_str}\n"
    "Answer: "
)
TEXT_QA_PROMPT = PromptTemplate(
    TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)
TEXT_QA_PROMPT_SEL = SelectorPromptTemplate(
    default_template=TEXT_QA_PROMPT,
    # conditionals=default_text_qa_conditionals,
)


class AppendReferenceSynthesizer(CompactAndRefine):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._text_qa_template = TEXT_QA_PROMPT_SEL
        self.verbose = kwargs.get("verbose")

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        verbose=None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get response."""
        if verbose is True or (verbose is None and self.verbose):
            logger.info(f"self._text_qa_template:\n{pformat(self._text_qa_template)}")
            logger.info(f"text_chunks:\n{pformat(text_chunks)}")
        return super().get_response(query_str, text_chunks)
        ...

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        verbose=None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get response."""
        if verbose is True or (verbose is None and self.verbose):
            logger.info(f"self._text_qa_template:\n{pformat(self._text_qa_template)}")
            logger.info(f"text_chunks:\n{pformat(text_chunks)}")
        return await super().aget_response(query_str, text_chunks)
