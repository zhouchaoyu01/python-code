import sqlite3
import os
from datetime import datetime
from langchain_chroma import Chroma
from .model_factory import ModelFactory
from .config import settings
from utils.logger import setup_logger



logger = setup_logger("VectorManager")
class VectorManager:
    def __init__(self):
        self.embeddings = ModelFactory.get_embedding()
        self.vector_store = Chroma(
            collection_name=settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=settings.chroma_persist_dir
        )
        # 初始化 SQLite 账本
        self.db_path = os.path.join(settings.chroma_persist_dir, "file_registry.db")
        self._init_metadata_db()
        logger.info(f"向量库初始化成功，存储路径: {settings.chroma_persist_dir}")

    def _init_metadata_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS file_records 
                (file_hash TEXT PRIMARY KEY, file_name TEXT, upload_time TEXT, chunk_count INTEGER)''')

    def add_documents(self, documents: list, file_info: dict, batch_size: int = 100):
        """增强版入库：记录哈希元数据"""
        file_hash = file_info['file_hash']
        logger.info(f"开始处理文件入库: {file_info['file_name']}, Hash: {file_hash}")


        try:
            # 写入前先清理旧哈希（幂等性）
            self.vector_store.delete(where={"file_hash": file_hash})
            
            for doc in documents:
                doc.metadata["file_hash"] = file_hash
                doc.metadata["file_name"] = file_info['file_name']

            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                self.vector_store.add_documents(batch)
                
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO file_records VALUES (?, ?, ?, ?)",
                    (file_hash, file_info['file_name'], 
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(documents)))
            logger.info(f"文件 {file_info['file_name']} 入库成功，共 {len(documents)} 个切片")
        except Exception as e:
            logger.error(f"文件 {file_info['file_name']} 入库失败: {str(e)}")
            raise e

    def get_file_list(self):
        """获取已上传文件列表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM file_records ORDER BY upload_time DESC")
            return [dict(row) for row in cursor.fetchall()]

    def delete_file_by_hash(self, file_hash: str):
        """双删逻辑"""
        self.vector_store.delete(where={"file_hash": file_hash})
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM file_records WHERE file_hash = ?", (file_hash,))

    def get_retriever(self):
        logger.info(f"获取检索器，K={settings.top_k}")
        return self.vector_store.as_retriever(
            # search_type="similarity_score_threshold",
             search_type="similarity", # 暂时不用 threshold 模式
            search_kwargs={"k": settings.top_k}
        )