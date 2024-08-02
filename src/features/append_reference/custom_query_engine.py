from llama_index.core.query_engine.retriever_query_engine import (
    RetrieverQueryEngine,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.schema import QueryType
from llama_index.core.base.response.schema import RESPONSE_TYPE


class ManualAppendReferenceQueryEngine(RetrieverQueryEngine):
    """RAG Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer

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
        sources_fmt = self._compile_sources(response_obj)
        if sources_fmt:
            response_fmt = f"""
{response_fmt}

{sources_fmt}
"""
        paragraphs_fmt = self._compile_ref_paragraphs(response_obj)
        if paragraphs_fmt:
            response_fmt = f"""
{response_fmt}

{paragraphs_fmt}
"""

        response_obj.response = response_fmt

        return response_obj

    def _compile_sources(self, response: RESPONSE_TYPE) -> str:
        sources = dict()
        for node in response.source_nodes:
            url = node.metadata.get("url")
            title = node.metadata.get("title")
            if url not in sources:
                sources[url] = title
        if sources:
            refs = [f"[{title}]({url})" for url, title in sources.items()]
            refs_fmt = ""
            for ref in refs:
                refs_fmt += f"- {ref}\n"
            output = f"""
Sources:
{refs_fmt}
"""
            return output

    def _compile_ref_paragraphs(
        self, response: RESPONSE_TYPE, score_threshold: float = 0.6
    ) -> str:
        paragraphs = dict()
        for node in response.source_nodes:
            title = node.metadata.get("title")
            if node.score < score_threshold:
                continue
            # Filter by score
            if title not in paragraphs:
                paragraphs[title] = [node.text]
            else:
                paragraphs[title].append(node.text)
        if paragraphs:
            paragraphs_fmt = ""
            for title, paragraphs in paragraphs.items():
                paragraphs_fmt += f"Article: **{title}**"
                for paragraph in paragraphs:
                    # Current workaround for visibility is to truncate the paragraph
                    # TODO: Find a better implementation
                    paragraphs_fmt += f"\n\n> ...{paragraph[:1000]}..."
                paragraphs_fmt += "\n\n"
            if paragraphs_fmt:
                output = f"""
#### Referenced Paragraphs
{paragraphs_fmt}
    """
                return output
