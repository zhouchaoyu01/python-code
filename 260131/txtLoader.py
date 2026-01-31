from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader(file_path="./data/0131.txt", encoding="utf-8")

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separators=["\n\n", "\n", " ", "","。","?","！"],
    length_function=len
)

texts = text_splitter.split_documents(data)

print(len(texts))

for i, t in enumerate(texts):
    print(f"--- 第 {i+1} 段 ---")
    print(t.page_content)