from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.schema import QueryType


class ManualAppendReferenceQueryEngine(RetrieverQueryEngine):
    """RAG Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer

    def __init__(self, ref_score_threshold: float = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_score_threshold = ref_score_threshold

    def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        response_obj = super().query(str_or_query_bundle)
        return self._append_sources_to_response_obj(response_obj)

    async def aquery(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        response_obj = await super().aquery(str_or_query_bundle)
        return self._append_sources_to_response_obj(response_obj)

    def _append_sources_to_response_obj(
        self, response_obj: RESPONSE_TYPE
    ) -> RESPONSE_TYPE:
        response_fmt = response_obj.response
        paragraphs_fmt = self._compile_ref_paragraphs(response_obj)
        if paragraphs_fmt:
            response_fmt = f"""
{response_fmt}

{paragraphs_fmt}
"""

        response_obj.response = response_fmt

        return response_obj

    def _compile_ref_paragraphs(self, response: RESPONSE_TYPE) -> str:
        paragraphs = dict()
        for node in response.source_nodes:
            biz_name = node.metadata.get("biz_name")
            review_id = node.metadata.get("review_id")
            if self.ref_score_threshold and node.score < self.ref_score_threshold:
                continue
            ref_dict = dict(review_id=review_id, text=node.text)
            # Filter by score
            if biz_name not in paragraphs:
                paragraphs[biz_name] = [ref_dict]
            else:
                paragraphs[biz_name].append(ref_dict)
        if paragraphs:
            paragraphs_fmt = ""
            for biz_name, paragraphs in paragraphs.items():
                paragraphs_fmt += f"**{biz_name}**"
                for ref_dict in paragraphs:
                    paragraph = (
                        f"{ref_dict['text']} (Review: `{ref_dict['review_id']}`)"
                    )
                    TRUNCATE_TRHESHOLD_CHARS = 1000
                    # Current workaround for visibility is to truncate the paragraph
                    # TODO: Find a better implementation
                    if len(paragraph) >= TRUNCATE_TRHESHOLD_CHARS:
                        paragraphs_fmt += f"\n\n> ...{paragraph[:1000]}..."
                    else:
                        paragraphs_fmt += f"\n\n> {paragraph}"
                paragraphs_fmt += "\n\n"
            if paragraphs_fmt:
                output = f"""
#### Referenced Paragraphs
{paragraphs_fmt}
    """
                return output
