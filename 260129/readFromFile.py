from langchain_core.messages import message_to_dict,messages_from_dict,BaseMessage
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
import os,json
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate   ,ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory

class FileChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores messages in a file."""

    def __init__(self, session_id: str, store_path: str):
        """Initialize with file path."""
        self.session_id = session_id
        self.store_path = store_path
        self.file_path = os.path.join(self.store_path, self.session_id)
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_message(self, message: BaseMessage) -> None:
        """Add a single message to the file (LangChain-compatible API).

        The framework expects `history.messages` to exist and `add_message`
        to accept a single `BaseMessage`. Persist the full message list
        after appending the new message.
        """
        msgs = list(self.messages)
        msgs.append(message)
        serialized = [message_to_dict(m) for m in msgs]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Optional helper: add multiple messages at once."""
        msgs = list(self.messages)
        msgs.extend(messages)
        serialized = [message_to_dict(m) for m in msgs]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False)

    @property
    def messages(self) -> list[BaseMessage]:
        """Return all messages as a list (required by RunnableWithMessageHistory).

        This property returns a plain list so callers can do `hist.messages.copy()`.
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)
                return messages_from_dict(messages_data)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([],f)





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
    return FileChatMessageHistory(session_id, "./chat_history/")



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

 
    response = conversation_chain.invoke({"input": inputs[0]}, session_config)
    print(response)
        