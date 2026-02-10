
from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool

@tool(description="获取天气信息")
def get_Weather()->str:
    return "明天上海天气晴朗，最高温度25度，最低温度15度。"



agent = create_agent(
    model=ChatTongyi(model = "qwen3-max"),
    tools=[get_Weather],
    system_prompt="你是一个聊天助手，可以回答用户问题。",
)


res = agent.invoke(
    {
        "messages":[
            {"role": "user", "content": "明天上海天气怎么样？"}
        ]
    }
)

# print(res)
"""
{'messages': [HumanMessage(content='明天上海天气怎么样？', additional_kwargs={}, response_metadata={}, id='c7ea8589-2d1f-43ae-b7ba-83ec6daf3b84'), AIMessage(content='', additional_kwargs={'tool_calls': [{'function': {'arguments': '{}', 'name': 'get_Weather'}, 'id': 'call_b231b4add726440a97203168', 'index': 0, 'type': 'function'}]}, response_metadata={'model_name': 'qwen3-max', 'finish_reason': 'tool_calls', 'request_id': '1a001425-6d87-4b01-8002-7a261a520d93', 'token_usage': 
{'input_tokens': 251, 'output_tokens': 12, 'prompt_tokens_details': {'cached_tokens': 0}, 'total_tokens': 263}}, id='lc_run--019c47b3-76f3-75a3-aa94-af67a2e10e68-0', tool_calls=[{'name': 'get_Weather', 'args': {}, 'id': 'call_b231b4add726440a97203168', 'type': 'tool_call'}], invalid_tool_calls=[]), ToolMessage(content='明天上海天气晴朗，最高温度25度，最低温度15度。', name='get_Weather', id='678e0d68-0c15-44b4-b489-071e0b2aac68', tool_call_id='call_b231b4add726440a97203168'), 
AIMessage(content='明天上海天气晴朗，最高温度25度，最低温度15度。建议适当增减衣物，享受好天气！', additional_kwargs={}, response_metadata={'model_name': 'qwen3-max', 'finish_reason': 'stop', 'request_id': '766a7d1e-a867-418b-a6b5-2dbfa3563a8f', 'token_usage': {'input_tokens': 295, 'output_tokens': 28, 'prompt_tokens_details': {'cached_tokens': 0}, 'total_tokens': 323}}, id='lc_run--019c47b3-7bf6-7ed2-a4b9-8ac9e3ca19d5-0', tool_calls=[], invalid_tool_calls=[])]}


"""

for message in res["messages"]:
    print(type(message).__name__, message.content)

"""
HumanMessage 明天上海天气怎么样？
AIMessage 
ToolMessage 明天上海天气晴朗，最高温度25度，最低温度15度。
AIMessage 明天上海天气晴朗，最高温度25度，最低温度15度。建议根据温差适当增减衣物，享受好天气！

"""