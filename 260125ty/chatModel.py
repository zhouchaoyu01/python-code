from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


ChatTongyiExample = ChatTongyi(model="qwen-plus")  # Initialize the ChatTongyi model

messages = [
    # SystemMessage(content="你是一个边塞诗人"),
    # AIMessage(content="锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。"),
    # HumanMessage(content="写一首关于冬天的诗"),
    ('system', "你是一个边塞诗人"),
    ('ai', "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。"),
    ('human', "写一首关于冬天的诗"),
]

ChatTongyiExample.stream(input=messages)  # Stream the response

for chunk in ChatTongyiExample.stream(input=messages):
    print(chunk.content, end="", flush=True)  # Print each chunk as it arrives