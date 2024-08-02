MANUAL_EVAL_QA = [
    (
        "What are key features of llama-agents?",
        """
Key features of llama-agents are:
1. Distributed Service Oriented Architecture: every agent in LlamaIndex can be its own independently running microservice, orchestrated by a fully customizable LLM-powered control plane that routes and distributes tasks.
2. Communication via standardized API interfaces: interface between agents using a central control plane orchestrator. Pass messages between agents using a message queue.
3. Define agentic and explicit orchestration flows: developers have the flexibility to directly define the sequence of interactions between agents, or leave it up to an “agentic orchestrator” that decides which agents are relevant to the task.
4. Ease of deployment: launch, scale and monitor each agent and your control plane independently.
5. Scalability and resource management: use our built-in observability tools to monitor the quality and performance of the system and each individual agent service
""",
    ),
    (
        "What are the two critical areas of RAG system performance that are assessed in the 'Evaluating RAG with LlamaIndex' section of the OpenAI Cookbook?",
        """
Retrieval System and Response Generation.
""",
    ),
    (
        "What are the two main metrics used to evaluate the performance of the different rerankers in the RAG system?",
        """
Hit rate and Mean Reciprocal Rank (MRR)

Hit Rate: Hit rate calculates the fraction of queries where the correct answer is found within the top-k retrieved documents. In simpler terms, it’s about how often our system gets it right within the top few guesses.

Mean Reciprocal Rank (MRR): For each query, MRR evaluates the system’s accuracy by looking at the rank of the highest-placed relevant document. Specifically, it’s the average of the reciprocals of these ranks across all the queries. So, if the first relevant document is the top result, the reciprocal rank is 1; if it’s second, the reciprocal rank is 1/2, and so on.
""",
    ),
    # Below question is hard because LLM needs to follow the URL in the blog to get the information to answer
    (
        "How does the MemoryCache project by Mozilla utilize PrivateGPT_AI and LlamaIndex to enhance personal knowledge management while maintaining privacy? Provide a brief overview of the project and its key features.",
        """
The MemoryCache project by Mozilla aims to transform local desktop environments into on-device AI agents, utilizing PrivateGPT_AI and LlamaIndex to enhance personal knowledge management. It saves browser history and other local files to the user’s machine, allowing a local AI model to ingest and augment responses. This approach maintains privacy by avoiding cloud-based processing, focusing instead on generating insights from personal data. The project emphasizes creating a personalized AI experience that mirrors the original vision of personal computers as companions for thought.
""",
    ),
]
