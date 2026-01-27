from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

model = ChatTongyi(model="qwen-max")
prompt_temp = PromptTemplate.from_template("我邻居姓：{lastname} 名：{firstname}")


parser = StrOutputParser()
chain = prompt_temp | model | parser | model

res = chain.invoke({"lastname": "张", "firstname": "三"})
print(res.content)