from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    data_dir: str = "data"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k_dense: int = 8
    top_k_sparse: int = 8
    top_k_final: int = 4
    max_upload_mb: int = 20
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
