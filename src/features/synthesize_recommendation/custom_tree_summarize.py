from llama_index.core.prompts import SelectorPromptTemplate
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.prompts.utils import is_chat_model

CUSTOM_TREE_SUMMARIZE_TMPL = (
    "Context information about various places to visit is provided below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Based on only the provided information, recommend multiple places to visit that match the user's preferences. "
    "You should rank the recommendations based on how relevant they are to the user's query"
    "Provide a summary explanation of the strengths of each option and compare them with each other based on different intentions.\n"
    "User Query: {query_str}\n"
    "Recommendations: "
)
CUSTOM_TREE_SUMMARIZE_PROMPT = PromptTemplate(
    CUSTOM_TREE_SUMMARIZE_TMPL, prompt_type=PromptType.SUMMARY
)

# Tree Summarize
# TREE_SUMMARIZE_PROMPT_TMPL_MSGS = [
#     TEXT_QA_SYSTEM_PROMPT,
#     ChatMessage(
#         content=(
#             "Context information from multiple sources is below.\n"
#             "---------------------\n"
#             "{context_str}\n"
#             "---------------------\n"
#             "Given the information from multiple sources and not prior knowledge, "
#             "answer the query.\n"
#             "Query: {query_str}\n"
#             "Answer: "
#         ),
#         role=MessageRole.USER,
#     ),
# ]

# CHAT_TREE_SUMMARIZE_PROMPT = ChatPromptTemplate(
#     message_templates=TREE_SUMMARIZE_PROMPT_TMPL_MSGS


# default_tree_summarize_conditionals = [(is_chat_model, CHAT_TREE_SUMMARIZE_PROMPT)]

CUSTOM_TREE_SUMMARIZE_PROMPT_SEL = SelectorPromptTemplate(
    default_template=CUSTOM_TREE_SUMMARIZE_PROMPT,
    # conditionals=default_tree_summarize_conditionals,
)
