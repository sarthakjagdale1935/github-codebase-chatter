# GitHub Codebase Chatter (Code-Aware RAG)

A Code-Aware Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Weaviate**, **Google Gemini**, and **Streamlit**.

## Features
- **AST & Syntax-Aware Chunking**: Splits source code along class, function, and interface boundaries using LangChain language splitters.
- **Vector Search with File Attribution**: Stores code chunks in Weaviate Cloud with exact repository file path metadata.
- **Interactive Architectural Q&A**: Ask structural or implementation questions about any public GitHub repository.

## Streamlit Community Cloud Secrets
Add these three keys under **Advanced settings > Secrets** (in TOML format):
```toml
GOOGLE_API_KEY = "your_gemini_api_key"
WEAVIATE_URL = "your_weaviate_cluster_url"
WEAVIATE_API_KEY = "your_weaviate_api_key"
```
