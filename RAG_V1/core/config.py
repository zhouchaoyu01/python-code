from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # 自动映射环境变量 DASHSCOPE_API_KEY，若不存在则报错
    # 生产环境下建议通过 export DASHSCOPE_API_KEY=xxx 注入
    dashscope_api_key: str = Field(..., env="DASHSCOPE_API_KEY")
    
    llm_model: str = "qwen-plus"
    embedding_model: str = "text-embedding-v2"
    
    chroma_persist_dir: str = "./ragv1/data/chroma_db"
    collection_name: str = "rag_collection"
    
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4

    # Pydantic v2 配置
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore" # 忽略多余的环境变量
    )

settings = Settings()