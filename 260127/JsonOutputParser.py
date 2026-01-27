from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

model = ChatTongyi(model="qwen-max")
first_prompt_temp = PromptTemplate.from_template("我邻居姓：{lastname} 性别：{gender},帮其取个姓名，并封装成Json格式给我，其中key为name,value对应取好的名字,严格按要求生成")
second_prompt_temp = PromptTemplate.from_template("解释{name}的含义")

parser1 = StrOutputParser()
parser2 = JsonOutputParser()
chain = first_prompt_temp | model | parser2  | second_prompt_temp  | model | parser1


res = chain.stream({"lastname": "张", "gender": "男"})
print(res)