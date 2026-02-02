from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .model_factory import ModelFactory
from .vector_manager import VectorManager

class RAGEngine:
    def __init__(self):
        self.llm = ModelFactory.get_llm()
        self.vector_manager = VectorManager()
        self.retriever = self.vector_manager.get_retriever()

    def _format_docs(self, docs):
        """增加去重逻辑"""
        seen_content = set()
        unique_docs = []
        for doc in docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc.page_content)
                seen_content.add(doc.page_content)
        
        # 如果检索不到任何内容，返回一个提示词而不是空字符串
        if not unique_docs:
            return "【暂无相关参考文档，请提示用户根据已知常识回答或补充知识库】"
        
        return "\n\n".join(unique_docs)

    def get_chain(self):
        """构建 RAG 链"""
        # 定义 Prompt 模板
        template = """你是一个专业的企业助手。请根据以下提供的上下文信息回答用户的问题。
如果你在上下文中找不到答案，请诚实地告诉用户你不知道，不要试图编造答案。

上下文内容:
{context}

用户问题: {question}

回答要求：请使用 Markdown 格式，回答简洁专业。"""


        base_retriever = self.vector_manager.get_retriever()
    

        prompt = ChatPromptTemplate.from_template(template)

        # 构建 LCEL 链
        rag_chain = (
            {"context": base_retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain