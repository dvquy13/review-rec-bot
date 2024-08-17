from llama_index.core.prompts.base import PromptTemplate

CUSTOM_CITATION_QA_TEMPLATE = PromptTemplate(
    "Based on only the provided information, recommend multiple places to visit that match the user's preferences. "
    "Include information about the places that would help the user make decisions, e.g. location and categories"
    "You should rank the recommendations based on how relevant they are to the user's query"
    "Provide a summary explanation of the strengths of each option and compare them with each other based on different intentions.\n"
    "When referencing information from a source review, "
    "cite the appropriate review(s) using their corresponding numbers. "
    "You must indicate that you cite a source by using the square bracket "
    "enclosing the source number, e.g [1] to denote you referring the source 1. "
    "Every answer should include at least one source citation. "
    "Only cite a source review when you are explicitly referencing it. "
    "If none of the sources are helpful, you should indicate that. "
    "<EXAMPLE>:\n"
    "Source 1:\n"
    "Cake Mix is my favorite. Great place to top off a date.\n"
    "Source 2:\n"
    "I'm looking forward to coming back next week. I can't believe on Thursday tall PBR drafts are 2.50.\n"
    "Query: What are some places to enjoy cake?\n"
    "Answer: Based on your query about places to enjoy cake, here are several recommendations ranked by relevance:"
    "### 1. Miha Kitchen:\n- Address: <placeholder>\n- Categories: <placeholder>\n- Summary: Miha Kitchen is highly praised for its delicious offerings, including cheese cake. "
    " The bakery has a cute space and a variety of grab-and-go options, especially for the Cake Mix [1]. "
    "The positive reviews highlight the quality of their food and drinks, particularly the chese cake, which is noted as good."
    "If you are considering a date to go there, try Thursday since it's said to have affordable beer [2]\n"
    'Sources:\n- [1]: "Cake Mix is my favorite. Great place to top off a date."\n'
    '- [2]: "I can\'t believe on Thursday tall PBR drafts are 2.50."\n'
    "</EXAMPLE>:\n"
    "IMPORTANT:\n-You must include all the cited source reviews you have used at the end of your answer\n"
    "Now it's your turn. Below are several numbered sources of information:"
    "\n------\n"
    "{context_str}"
    "\n------\n"
    "User Query: {query_str}\n"
    "Recommendations: "
)

CUSTOM_CITATION_REFINE_TEMPLATE = PromptTemplate(
    "Based on only the provided information, recommend multiple places to visit that match the user's preferences. "
    "Include information about the places that would help the user make decisions, e.g. location and categories"
    "You should rank the recommendations based on how relevant they are to the user's query"
    "Provide a summary explanation of the strengths of each option and compare them with each other based on different intentions.\n"
    "When referencing information from a source review, "
    "cite the appropriate review(s) using their corresponding numbers. "
    "You must indicate that you cite a source by using the square bracket "
    "enclosing the source number, e.g [1] to denote you referring the source 1. "
    "Every answer should include at least one source citation. "
    "Only cite a source review when you are explicitly referencing it. "
    "If none of the sources are helpful, you should indicate that. "
    "<EXAMPLE>:\n"
    "Source 1:\n"
    "Cake Mix is my favorite. Great place to top off a date.\n"
    "Source 2:\n"
    "I'm looking forward to coming back next week. I can't believe on Thursday tall PBR drafts are 2.50.\n"
    "Query: What are some places to enjoy cake?\n"
    "Answer: Based on your query about places to enjoy cake, here are several recommendations ranked by relevance:"
    "### 1. Miha Kitchen:\n- Address: <placeholder>\n- Categories: <placeholder>\n- Summary: Miha Kitchen is highly praised for its delicious offerings, including cheese cake. "
    " The bakery has a cute space and a variety of grab-and-go options, especially for the Cake Mix [1]. "
    "The positive reviews highlight the quality of their food and drinks, particularly the chese cake, which is noted as good."
    "If you are considering a date to go there, try Thursday since it's said to have affordable beer [2]\n"
    'Sources:\n- [1]: "Cake Mix is my favorite. Great place to top off a date."\n'
    '- [2]: "I can\'t believe on Thursday tall PBR drafts are 2.50."\n'
    "</EXAMPLE>:\n"
    "IMPORTANT:\n-You must include all the cited source reviews you have used at the end of your answer\n"
    "Now it's your turn."
    "We have provided an existing answer: {existing_answer}"
    "Below are several numbered sources of information. "
    "Use them to refine the existing answer. "
    "Please double check and make sure that the correct sources are quoted at the end. "
    "If the provided sources are not helpful, you will repeat the existing answer."
    "\nBegin refining!"
    "\n------\n"
    "{context_msg}"
    "\n------\n"
    "User Query: {query_str}\n"
    "Recommendations: "
)
