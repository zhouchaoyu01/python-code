import os
import logging
import time
from typing import List
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from core.vector_manager import VectorManager
from core.config import settings
from .hash_utils import calculate_file_hash # 引入新增工具
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        # 初始化切片器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            add_start_index=True, # 记录切片在原文中的位置，方便溯源
        )
        self.vector_manager = VectorManager()

    def load_file(self, file_path: str) -> List[Document]:
        """根据文件后缀选择不同的加载器"""
        ext = os.path.splitext(file_path)[-1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext in [".docx", ".doc"]:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif ext == ".txt":
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                logger.warning(f"暂不支持的文件格式: {ext}")
                return []
            docs = loader.load()
            for doc in docs:
                upload_time = time.time()
                # print(upload_time)
                doc.metadata["upload_time"] = upload_time # 记录上传时间
                doc.metadata["file_name"] = os.path.basename(file_path)
            return docs
        except Exception as e:
            logger.error(f"解析文件 {file_path} 出错: {e}")
            return []

    def process_directory(self, dir_path: str):
        """批量处理目录下所有文档"""
        path = Path(dir_path)
        if not path.exists():
            logger.error(f"目录不存在: {dir_path}")
            return

        all_docs = []
        # 支持递归查找
        for file in path.rglob("*.*"):
            if file.suffix in [".pdf", ".docx", ".txt"]:
                logger.info(f"正在解析: {file.name}")
                docs = self.load_file(str(file))
                all_docs.extend(docs)

        if not all_docs:
            logger.info("没有找到可处理的文档。")
            return

        # 执行切片
        logger.info(f"正在对 {len(all_docs)} 个文档进行切片...")
        splits = self.text_splitter.split_documents(all_docs)
        
        # 存入向量库
        logger.info(f"正在将 {len(splits)} 条片段存入向量库...")
        self.vector_manager.add_documents(splits)
        logger.info("文档入库完成！")

# 脚本独立运行入口
if __name__ == "__main__":
    # 使用示例：将 docs 文件夹下的文件全部入库
    processor = DocumentProcessor()
    processor.process_directory("./docs")