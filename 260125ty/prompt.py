from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi


prompt_template = PromptTemplate.from_template("Translate the following English texts to French: {text} {text2}")

model = Tongyi(model="qwen-max")  # Initialize the Tongyi model

# prompt_text = prompt_template.format(text="Hello, how are you?")  # Example of formatting the prompt with input text
# res = model.invoke(input=prompt_text)  # Invoke the model with the formatted prompt
# print(res)

chain = prompt_template | model
res = chain.invoke({"text": "Hello, how are you?", "text2": "What is your name?"})
print(res)

