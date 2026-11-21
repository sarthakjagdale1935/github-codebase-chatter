import os
import weaviate
from weaviate.auth import AuthApiKey
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are an expert principal software engineer analyzing a codebase.
Answer the developer's question using ONLY the retrieved code snippets provided below.
For each file or logic you cite, explicitly mention the file path (`file_path`).
If the retrieved context does not contain enough information to answer fully, state what is missing.

Retrieved Code Context:
{context}
"""


def get_embeddings():
    for model_name in ["text-embedding-004", "models/gemini-embedding-001"]:
        try:
            emb = GoogleGenerativeAIEmbeddings(model=model_name)
            emb.embed_query("test")
            return emb
        except Exception:
            continue
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def get_llm():
    for model_name in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
        try:
            return ChatGoogleGenerativeAI(model=model_name, temperature=0.1)
        except Exception:
            continue
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)


def ask_codebase(query: str, index_name: str = "CodebaseChunk"):
    client = None
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        )

        if not client.collections.exists(index_name):
            raise ValueError(
                f"Index '{index_name}' does not exist yet. Please clone and index a repository from the sidebar first."
            )

        embeddings = get_embeddings()

        vector_store = WeaviateVectorStore(
            client=client,
            index_name=index_name,
            text_key="text",
            embedding=embeddings,
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": 6})
        retrieved_docs = retriever.invoke(query)

        context_blocks = []
        sources = []
        for doc in retrieved_docs:
            file_path = doc.metadata.get("file_path", "Unknown File")
            lang = doc.metadata.get("language", "code")
            block = f"--- File: {file_path} ({lang}) ---\n{doc.page_content}\n"
            context_blocks.append(block)
            sources.append(
                {
                    "file_path": file_path,
                    "language": lang,
                    "snippet": doc.page_content,
                }
            )

        context_str = "\n".join(context_blocks)

        llm = get_llm()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )

        chain = prompt | llm
        response = chain.invoke({"context": context_str, "question": query})
        return response.content, sources

    finally:
        if client is not None:
            client.close()
