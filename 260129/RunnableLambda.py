from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate   
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

model = ChatTongyi(model="qwen3-max")

first_prompt_temp = PromptTemplate.from_template("我邻居姓：{lastname} 性别：{gender},帮其取个姓名，仅给我提供一个名字,严格按要求生成")
second_prompt_temp = PromptTemplate.from_template("解释{name}的含义")

my_func = RunnableLambda(lambda ai_msg :{"name": ai_msg.content}) 
chain = first_prompt_temp | model | my_func  | second_prompt_temp  | model | StrOutputParser()

for chunk in chain.stream({"lastname": "李", "gender": "女"}):
    print(chunk, end="",flush=True)