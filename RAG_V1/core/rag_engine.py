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
        """格式化检索到的文档，方便注入 Prompt"""
        return "\n\n".join(doc.page_content for doc in docs)

    def get_chain(self):
        """构建 RAG 链"""
        # 定义 Prompt 模板
        template = """你是一个专业的企业助手。请根据以下提供的上下文信息回答用户的问题。
如果你在上下文中找不到答案，请诚实地告诉用户你不知道，不要试图编造答案。

上下文内容:
{context}

用户问题: {question}

回答要求：请使用 Markdown 格式，回答简洁专业。"""

        prompt = ChatPromptTemplate.from_template(template)

        # 构建 LCEL 链
        rag_chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain