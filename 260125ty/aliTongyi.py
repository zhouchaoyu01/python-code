from langchain_community.llms.tongyi import Tongyi


model = Tongyi(model="qwen-max")

res = model.invoke(input="Hello, Tongyi!")

print(res)