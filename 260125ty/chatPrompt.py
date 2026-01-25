from langchain_core.prompts import ChatPromptTemplate  ,MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi 


model = ChatTongyi(model="qwen-max") 


chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', '你是一个有帮助的助手。'),
        MessagesPlaceholder("history"),
        ('human', '请告诉我今天出门需要准备什么？'),
    ]
)

history_data = [
    ('human','今天天气灰蒙蒙的，有好多乌云'),
    ('ai','是的，今天可能会下雨，建议带把伞。'),
    ('human', '那我还需要准备什么吗？'),
    ('ai','建议穿上防水的鞋子，避免淋湿。'),
]

prompt_text = chat_prompt_template.invoke({"history": history_data}).to_string()

print("=== Prompt Text ===")
print(prompt_text)

res = model.invoke(prompt_text)

print("=== Model Response ===")
print(res)