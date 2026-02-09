from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from .model_factory import ModelFactory
from .vector_manager import VectorManager
from operator import itemgetter # 引入这个工具，专门用于从字典取值
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from utils.logger import setup_logger


logger = setup_logger("RAGEngine")

def print_debug_prompt(prompt) -> ChatPromptTemplate:
    """调试用，打印最终发送给 LLM 的 Prompt"""
    logger = setup_logger("RAGEngine")
    logger.info("===== Prompt Start =====")
    logger.info(prompt.to_string())
    logger.info("===== Prompt End =====")
    return prompt

def debug_step(data, step_name: str):
    """自定义调试函数，打印数据内容和类型"""
    logger.info(f"==== [DEBUG: {step_name}] Start ====")
    # 打印数据类型
    logger.info(f"Type: {type(data)}")
    # 打印数据内容摘要
    if isinstance(data, dict):
        # 排除掉太长的 context 或 chat_history 以免刷屏
        debug_info = {k: (str(v)[:100] + "...") if isinstance(v, str) else str(v) for k, v in data.items()}
        logger.info(f"Content: {json.dumps(debug_info, ensure_ascii=False)}")
    else:
        logger.info(f"Content: {str(data)[:200]}...")
    logger.info(f"==== [DEBUG: {step_name}] End ====\n")
    return data


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
    def _format_docs_with_sources(self, docs):
        """
        格式化文档，并在内容前注入 [编号] 和 文件名
        """
        formatted_chunks = []
        for i, doc in enumerate(docs):
            source_name = doc.metadata.get("file_name", "未知文件")
            # 格式示例：[1] (来源: manual.pdf): 内容内容...
            content = f"[{i+1}] (来源: {source_name}):\n{doc.page_content}"
            formatted_chunks.append(content)
        
        if not formatted_chunks:
            return "【暂无参考资料】"
        
        return "\n\n".join(formatted_chunks)
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
        condense_question_chain = rephrase_prompt | self.llm | StrOutputParser() 


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
    

    def get_chain_with_source(self):
        
        
        # 1. 问题重写链 (保持不变)
        rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", "参考对话历史，将用户问题重写为独立的搜索查询。不要回答问题，只需重写。"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        condense_question_chain = rephrase_prompt| self.llm | StrOutputParser()| RunnableLambda(lambda x: debug_step(x, "重写后的问题"))
        logger.info(f"condense_question_chain: {condense_question_chain}")
        # 2. 增强版问答 Prompt
        # 明确要求 LLM 引用编号
        qa_system_prompt = (
            "你是一个专业的企业助手。请利用以下参考内容回答问题。\n"
            "回答要求：\n"
            "1. 必须根据参考内容回答，严禁编造。\n"
            "2. **必须在每个回答段落末尾标注引用的来源编号，例如 [1] 或 [1][2]**。\n"
            # "3. 如果参考内容中没有答案，请直说不知道。\n\n"
            "参考内容:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        # 3. 构建 LCEL 链
        # 我们使用一个 dict 包装结果，以便同时返回 answer 和 context(docs)
        def get_answer_and_sources(data):
            # 这个函数接收最终组装好的数据字典
            final_chain = qa_prompt | self.llm | StrOutputParser()| RunnableLambda(lambda x: debug_step(x, "最终回答"))
            return {
                "answer": (final_chain).invoke(data),
                "context": data["raw_docs"] # 将原始文档列表传出
            }

        full_rag_chain = (
            RunnablePassthrough.assign(
                standalone_question=condense_question_chain
            )
            | RunnablePassthrough.assign(
                # 这一步检索出 raw_docs 列表，并生成格式化的 context 字符串
                raw_docs=itemgetter("standalone_question") | self.retriever
            )
            | RunnableLambda(lambda x: debug_step(x, "检索后的完整字典")) # 观测检索结果
            | RunnablePassthrough.assign(
                context=lambda x: self._format_docs_with_sources(x["raw_docs"])
            )
            | RunnableLambda(get_answer_and_sources) # 最终产出字典
        )

        return RunnableWithMessageHistory(
            full_rag_chain,
            get_session_history, # 引用你代码中定义的 get_session_history
            input_messages_key="input",
            history_messages_key="chat_history",
            # 注意：由于输出是字典，这里不再配置 output_messages_key 以便透传
        )