from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool

@tool(description="获取股票价格，输入股票名称，返回当前价格字符串信息")
def get_price(name:str)->str:
    return f"{name}的当前价格是100元。"

@tool(description="获取股票信息，输入股票名称，返回该股票的相关信息字符串")
def get_info(name:str)->str:
    return f"{name}是上市公司。"


agent = create_agent(
    model=ChatTongyi(model = "qwen3-max"),
    tools=[get_price, get_info],
    system_prompt="你是一个智能助手，可以回答用户关于股票的问题。记得告诉我思考过程，让我知道你是如何得出结论的。",
)


for chunk in agent.stream(

    {
        "messages":[
            {"role": "user", "content": "请告诉我苹果公司的当前股票价格和相关信息。"}
        ]
    },
    stream_mode="values"
):
    # print(chunk)
    lastest_message = chunk["messages"][-1]
    if lastest_message.content:
        print(type(lastest_message).__name__, lastest_message.content)

    try:
        if lastest_message.tool_calls:
            print(f"工具调用：{[tool_call['name'] for tool_call in lastest_message.tool_calls]}")
    except AttributeError:
        pass

"""
{'messages': [HumanMessage(content='请告诉我苹果公司的当前股票价格和相关信息。', additional_kwargs={}, response_metadata={}, id='d7c925b8-6902-4d4a-a93c-177a9804a0ba')]}
{'messages': [HumanMessage(content='请告诉我苹果公司的当前股票价格和相关信息。', additional_kwargs={}, response_metadata={}, id='d7c925b8-6902-4d4a-a93c-177a9804a0ba'), AIMessage(content='', additional_kwargs={'tool_calls': [{'function': {'arguments': '{"name": "苹果公司"}', 'name': 'get_price'}, 'id': 'call_16157cfa507d4874b5cc53df', 'index': 0, 'type': 'function'}, {'function': {'arguments': '{"name": "苹果公司"}', 'name': 'get_info'}, 'id': 'call_8d4b3606a46f420eba8244c4', 'index': 1, 'type': 'function'}]}, response_metadata={'model_name': 'qwen3-max', 'finish_reason': 'tool_calls', 'request_id': '7dbc5cb6-2b33-47d6-a1c6-664029ad9c38', 'token_usage': {'input_tokens': 359, 'output_tokens': 43, 'prompt_tokens_details': {'cached_tokens': 0}, 'total_tokens': 402}}, id='lc_run--019c47c1-6bd6-7192-bb3f-417a653704ff-0', tool_calls=[{'name': 'get_price', 'args': {'name': '苹果公司'}, 'id': 'call_16157cfa507d4874b5cc53df', 'type': 'tool_call'}, {'name': 'get_info', 'args': {'name': '苹果公司'}, 'id': 'call_8d4b3606a46f420eba8244c4', 'type': 'tool_call'}], invalid_tool_calls=[])]}
{'messages': [HumanMessage(content='请告诉我苹果公司的当前股票价格和相关信息。', additional_kwargs={}, response_metadata={}, id='d7c925b8-6902-4d4a-a93c-177a9804a0ba'), AIMessage(content='', additional_kwargs={'tool_calls': [{'function': {'arguments': '{"name": "苹果公司"}', 'name': 'get_price'}, 'id': 'call_16157cfa507d4874b5cc53df', 'index': 0, 'type': 'function'}, {'function': {'arguments': '{"name": "苹果公司"}', 'name': 'get_info'}, 'id': 'call_8d4b3606a46f420eba8244c4', 'index': 1, 'type': 'function'}]}, response_metadata={'model_name': 'qwen3-max', 'finish_reason': 'tool_calls', 'request_id': '7dbc5cb6-2b33-47d6-a1c6-664029ad9c38', 'token_usage': {'input_tokens': 359, 'output_tokens': 43, 'prompt_tokens_details': {'cached_tokens': 0}, 'total_tokens': 402}}, id='lc_run--019c47c1-6bd6-7192-bb3f-417a653704ff-0', tool_calls=[{'name': 'get_price', 'args': {'name': '苹果公司'}, 'id': 'call_16157cfa507d4874b5cc53df', 'type': 'tool_call'}, {'name': 'get_info', 'args': {'name': '苹果公司'}, 'id': 'call_8d4b3606a46f420eba8244c4', 'type': 'tool_call'}], invalid_tool_calls=[]), ToolMessage(content='苹果公司 
的当前价格是100元。', name='get_price', id='1d43d2e7-e488-488d-87a5-e5cb132ba67f', tool_call_id='call_16157cfa507d4874b5cc53df'), ToolMessage(content='苹果公司 
是上市公司。', name='get_info', id='65d178a6-e3f7-4d54-9088-4e196501474e', tool_call_id='call_8d4b3606a46f420eba8244c4')]}
{'messages': [HumanMessage(content='请告诉我苹果公司的当前股票价格和相关信息。', additional_kwargs={}, response_metadata={}, id='d7c925b8-6902-4d4a-a93c-177a9804a0ba'), AIMessage(content='', additional_kwargs={'tool_calls': [{'function': {'arguments': '{"name": "苹果公司"}', 'name': 'get_price'}, 'id': 'call_16157cfa507d4874b5cc53df', 'index': 0, 'type': 'function'}, {'function': {'arguments': '{"name": "苹果公司"}', 'name': 'get_info'}, 'id': 'call_8d4b3606a46f420eba8244c4', 'index': 1, 'type': 'function'}]}, response_metadata={'model_name': 'qwen3-max', 'finish_reason': 'tool_calls', 'request_id': '7dbc5cb6-2b33-47d6-a1c6-664029ad9c38', 'token_usage': {'input_tokens': 359, 'output_tokens': 43, 'prompt_tokens_details': {'cached_tokens': 0}, 'total_tokens': 402}}, id='lc_run--019c47c1-6bd6-7192-bb3f-417a653704ff-0', tool_calls=[{'name': 'get_price', 'args': {'name': '苹果公司'}, 'id': 'call_16157cfa507d4874b5cc53df', 'type': 'tool_call'}, {'name': 'get_info', 'args': {'name': '苹果公司'}, 'id': 'call_8d4b3606a46f420eba8244c4', 'type': 'tool_call'}], invalid_tool_calls=[]), ToolMessage(content='苹果公司 
的当前价格是100元。', name='get_price', id='1d43d2e7-e488-488d-87a5-e5cb132ba67f', tool_call_id='call_16157cfa507d4874b5cc53df'), ToolMessage(content='苹果公司 
是上市公司。', name='get_info', id='65d178a6-e3f7-4d54-9088-4e196501474e', tool_call_id='call_8d4b3606a46f420eba8244c4'), AIMessage(content='苹果公司的当前股票 
价格是100元。此外，苹果公司是一家上市公司。如果您需要更多详细信息或有其他问题，请随时告诉我！', additional_kwargs={}, response_metadata={'model_name': 'qwen3-max', 'finish_reason': 'stop', 'request_id': '10ad52f7-55ba-48d9-ad37-bf4fddced0a2', 'token_usage': {'input_tokens': 435, 'output_tokens': 31, 'prompt_tokens_details': {'cached_tokens': 0}, 'total_tokens': 466}}, id='lc_run--019c47c1-7508-7b62-b49e-79257dbeb717-0', tool_calls=[], invalid_tool_calls=[])]}




HumanMessage 请告诉我苹果公司的当前股票价格和相关信息。
工具调用：['get_price', 'get_info']
ToolMessage 苹果公司是上市公司。
AIMessage 苹果公司的当前股票价格是100元。此外，苹果公司是一家上市公司。如果您需要更详细的信息或有其他问题，请随时告诉我！
"""