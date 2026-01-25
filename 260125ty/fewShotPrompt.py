from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.llms.tongyi import Tongyi


model = Tongyi(model="qwen-max")  # Initialize the Tongyi model

example_template = PromptTemplate.from_template("单词：{word}, 反义词：{antonym}")

example_data = [
    {"word": "高兴", "antonym": "难过"},
    {"word": "快", "antonym": "慢"},
]

few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_template,
    examples=example_data,
    prefix="给出以下单词的反义词,有如下例子：",
    suffix="单词：{input_word} 反义词：",
    input_variables=["input_word"],
)


few_shot_prompt_text = few_shot_prompt.invoke(input={"input_word": "聪明"})  # Format the prompt with the input word

print("Formatted Few-Shot Prompt:")
print(few_shot_prompt_text)

chain = few_shot_prompt | model


chain_res = chain.invoke({"input_word": "勇敢"})  # Invoke the chain with the input word
print("Model Response:")
print(chain_res)