from langchain_community.vectorstores import Chroma
from .model_factory import ModelFactory
from .config import settings

class VectorManager:
    def __init__(self):
        self.embeddings = ModelFactory.get_embedding()
        self.vector_store = Chroma(
            collection_name=settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=settings.chroma_persist_dir
        )

    def add_documents(self, documents: list, batch_size: int = 100):
        """分批次存入，提高稳定性"""
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            self.vector_store.add_documents(batch)
            
    def get_retriever(self):
         # similarity_score_threshold: 仅返回相似度高于阈值的片段
        return self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": settings.top_k,
                "score_threshold": 0.5  # 建议根据测试在 0.4-0.6 之间调整
            }
    )