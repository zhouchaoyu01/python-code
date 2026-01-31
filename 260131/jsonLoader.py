from langchain_community.document_loaders import JSONLoader

"""
JSONLoader 依赖 jq 库，通过 pip install jq 安装。

JSONLoader 使用 jq 的解析语法，常见如：
. 表示根、[] 表示数组
.name 表示从根取 name 的值
.hobby[1] 表示取 hobby 对应数组的第二个元素
.[] 表示将数组内的每个字典 (JSON对象) 都取到
.[].name 表示取数组内每个字典 (JSON) 对象的 name 对应的值
JSONLoader 初始化有 4 个主要参数：

file_path: 文件路径，必填
jq_schema: jq 解析语法，必填
text_content: 提取到的是否是字符串，默认 True，非必填
json_lines: 是否是 JsonLines 文件，默认 False，非必填
JsonLines 文件：每一行都是一个独立的字典 (Json对象)

"""


# loader = JSONLoader(file_path="./data/0131.json"
#                     ,jq_schema=".array[0].file_path")
loader = JSONLoader(file_path="./data/0131.json"
                    ,jq_schema=".",
                    text_content=False)

data = loader.load()
print(data)