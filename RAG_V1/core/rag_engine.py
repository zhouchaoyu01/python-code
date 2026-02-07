from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from .model_factory import ModelFactory
from .vector_manager import VectorManager
from operator import itemgetter # 引入这个工具，专门用于从字典取值


from utils.logger import setup_logger
logger = setup_logger("RAGEngine")

def print_debug_prompt(prompt) -> ChatPromptTemplate:
    """调试用，打印最终发送给 LLM 的 Prompt"""
    logger = setup_logger("RAGEngine")
    logger.info("===== Prompt Start =====")
    logger.info(prompt.to_string())
    logger.info("===== Prompt End =====")
    return prompt

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    # 滑动窗口：保留最近 10 条（约 5 轮对话）
    if len(store[session_id].messages) > 10:
        recent = store[session_id].messages[-10:]
        store[session_id].clear()
        store[session_id].add_messages(recent)
    return store[session_id]

class RAGEngine:
    def __init__(self):
        self.llm = ModelFactory.get_llm()
        self.vector_manager = VectorManager()
        self.retriever = self.vector_manager.get_retriever()

    def _format_docs(self, docs):
        """保持原有的去重与空结果处理逻辑"""
        # 增加打印，方便在控制台调试
        logger.info(f"--- 向量检索完成，命中数量: {len(docs)} ---")
        seen = set()
        unique_docs = []
        for i, doc in enumerate(docs):
             # 记录每条命中的内容预览和分值（如果有）
            logger.info(f"命中片段 [{i}] 来源: {doc.metadata.get('file_name')} | 内容: {doc.page_content[:50]}...")
            if doc.page_content not in seen:
                unique_docs.append(doc.page_content)
                seen.add(doc.page_content)
        if not unique_docs:
            logger.warning("检索结果为空！可能是由于相似度阈值过滤了所有结果。")
            return "【暂无相关参考文档，请提示用户根据已知常识回答】"
        return "\n\n".join(unique_docs)
    
    def get_chain(self):
        """
        使用 LCEL 构建 1.0 风格的 RAG 链
        """
        def log_rephrased_question(data):
            logger.info(f"对话历史改写后的独立问题: {data}")
            return data
        # 1. 问题重写子链 (Condense Question Chain)
        # 作用：把 (chat_history + input) -> 转换为独立的问题
        rephrase_system_prompt = (
            "给定聊天记录和用户最新的问题，"
            "将其重写为一个可以独立理解的问题。不要回答问题，"
            "只需重写，如果不需要重写则直接返回原始问题。"
        )
        rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", rephrase_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        # 这是一个微型的 LCEL 链：Prompt -> LLM -> String
        condense_question_chain = rephrase_prompt |RunnableLambda(print_debug_prompt)| self.llm | StrOutputParser() 


        # 2. 最终问答子链 (Answer Generation Chain)
        qa_system_prompt = (
            "你是一个专业的企业助手。请利用以下参考内容和聊天历史回答问题。"
            "如果你不知道答案，就直说不知道。请保持回答简洁专业。"
            "\n\n参考内容:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        # 3. 完整 LCEL 组合逻辑
        # 关键点：使用 RunnablePassthrough.assign 动态构建上下文数据流
        full_rag_chain = (
            RunnablePassthrough.assign(
                # 第一步：先通过重写链得到独立问题
                standalone_question=condense_question_chain
            )
            | RunnableLambda(log_rephrased_question) # 记录中间结果
            | RunnablePassthrough.assign(
                # 第二步：用重写后的问题去检索文档，并格式化
                 context=itemgetter("standalone_question") | self.retriever | self._format_docs
            )
            | qa_prompt  # 第三步：将所有数据喂给问答 Prompt
            | RunnableLambda(print_debug_prompt)
            | self.llm   # 第四步：调用 LLM
            | StrOutputParser() # 第五步：解析输出
        )

        # 4. 封装记忆组件
        # 注意：RunnableWithMessageHistory 要求 input 和 chat_history 键名匹配
        return RunnableWithMessageHistory(
            full_rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )