from langchain_community.document_loaders import PyPDFLoader



loader = PyPDFLoader(file_path="./data/0131.pdf",mode="page")

i =0 
for doc in loader.lazy_load():
    i += 1
    print(f"--- 第 {i} 页 ---")
    print(doc.page_content)