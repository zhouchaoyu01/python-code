from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate   ,ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory


model = ChatTongyi(model="qwen3-max")
# prompt = PromptTemplate.from_template("你需要根据会话历史回答用户问题。对话历史：{chat_history} 用户提问：{input},请给出简洁的回答。")


prompt = ChatPromptTemplate.from_messages([

    ("system", "你需要根据会话历史回答用户问题。"),
    MessagesPlaceholder("chat_history"),
    ("user", "用户提问：{input},请给出简洁的回答。"),

])
str_parser = StrOutputParser()

base_chain = prompt | model | str_parser

store = {}

def get_history(session_id: str) :
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]



conversation_chain = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


if __name__ == "__main__":
    session_config = {
        "configurable": {
            "session_id": "112233",
        }
    }

    inputs = [
        "你好！",
        "你是谁？",
        "你能做什么？",
        "今天天气怎么样？",
    ]

    for user_input in inputs:
        response = conversation_chain.invoke(
            {"input": user_input}, session_config
        )
        print("AI:", response)