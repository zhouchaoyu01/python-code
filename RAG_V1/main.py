import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.concurrency import run_in_threadpool
import uuid
from langchain_core.runnables import RunnableLambda
from core.rag_engine import RAGEngine
from core.vector_manager import VectorManager
from utils.document_processor import DocumentProcessor
from utils.hash_utils import calculate_file_hash
from pathlib import Path  # 引入 Path 处理路径
# 初始化 FastAPI 应用
app = FastAPI(
    title="企业级 RAG 助手 API",
    description="基于 LangChain + 阿里百炼 + ChromaDB 的 RAG 基础架构",
    version="1.0.0"
)

# 初始化业务组件
# 注意：在实际生产中，建议使用依赖注入或在 app 启动事件中初始化
rag_engine = RAGEngine()
doc_processor = DocumentProcessor()

# --- 数据模型定义 ---
class ChatRequest(BaseModel):
    query: str
    session_id: str="default_session"

class ChatResponse(BaseModel):
    answer: str
    status: str

# --- 接口定义 ---

@app.get("/")
async def root():
    return {"message": "RAG API 运行中", "docs_url": "/docs"}

@app.post("/upload", summary="上传文档并自动入库")
async def upload_document(file: UploadFile = File(...)):
    """
    接收上传的文件(PDF/DOCX/TXT)，保存到临时目录，解析并存入向量库。
    """
     # 1. 确定基础目录（使用绝对路径更稳健）
    # Path(__file__).parent 拿到 main.py 所在的文件夹
    base_dir = Path(__file__).parent.resolve()
    temp_dir = base_dir/"temp_uploads"
    
    # 2. 确保目录存在（手动检查并创建）
    try:
        if not temp_dir.exists():
            temp_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"无法创建临时目录: {str(e)}")
    unique_filename = f"{uuid.uuid4()}{file.filename}"
     # 3. 生成唯一文件名，拼接完整文件路径
    file_path = temp_dir / unique_filename
    print(f"Saving uploaded file to: {file_path}")


    try:
        # 1. 保存文件到本地
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # # 2. 调用文档处理器进行解析和入库
        # # 这里为了演示方便，直接调用解析单文件的逻辑
        # docs = doc_processor.load_file(file_path)
        # if not docs:
        #     raise HTTPException(status_code=400, detail="文件解析失败或格式不支持")
        
        # splits = doc_processor.text_splitter.split_documents(docs)
        # doc_processor.vector_manager.add_documents(splits)

        # return {"message": f"文件 {file.filename} 已成功处理并入库", "chunks": len(splits)}

        print("文件保存成功，开始处理...")
        # 将耗时的同步任务交给线程池处理，防止阻塞 API
        def process_task():
            file_hash = calculate_file_hash(file_path)
            docs = doc_processor.load_file(file_path)
            splits = doc_processor.text_splitter.split_documents(docs)
            doc_processor.vector_manager.add_documents(splits, {"file_hash": file_hash, "file_name": file.filename})
            return len(splits)

        chunks_count = await run_in_threadpool(process_task)
        return {"status": "success", "chunks": chunks_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传处理失败: {str(e)}")
    finally:
        # 清理临时文件（可选）
        if os.path.exists(file_path):
            os.remove(file_path)
@app.get("/files")
async def list_files():
    return doc_processor.vector_manager.get_file_list()

@app.delete("/files/{file_hash}")
async def delete_file(file_hash: str):
    doc_processor.vector_manager.delete_file_by_hash(file_hash)
    return {"status": "success"}


@app.post("/chat", response_model=ChatResponse, summary="RAG 问答对话")
async def chat(request: ChatRequest):
    """
    输入问题，检索向量库，生成答案。
    """
    try:
        chain = rag_engine.get_chain()
        # 使用 ainvoke 进行异步调用
        response = await chain.ainvoke({"input": request.query}, config={"configurable": {"session_id": request.session_id}})
        return {"answer": response, "status": "success"}
    except Exception as e:
        # 打印详细日志方便调试
        print(f"Chat Error: {e}")
        import traceback
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=str(e))

# --- 启动配置 ---
if __name__ == "__main__":
    import uvicorn
    # host="127.0.0.1" 仅限本地访问，"0.0.0.0" 允许外网/局域网访问
    uvicorn.run(app, host="127.0.0.1", port=8000)