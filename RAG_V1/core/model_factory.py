import dashscope
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from .config import settings

class ModelFactory:
    """模型工厂类，负责初始化 LLM 和 Embedding 模型"""
    
    @staticmethod
    def get_llm():
        return ChatTongyi(
            model=settings.llm_model,
            dashscope_api_key=settings.dashscope_api_key,
            streaming=True,  # 支持流式输出
            temperature=0.7
        )

    @staticmethod
    def get_embedding():
        return DashScopeEmbeddings(
            model=settings.embedding_model,
            dashscope_api_key=settings.dashscope_api_key
        )