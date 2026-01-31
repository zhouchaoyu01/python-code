from langchain_community.document_loaders import CSVLoader


loader = CSVLoader(file_path="./data/0131.csv", encoding="utf-8", 
                   csv_args={
                       "delimiter": ",", "quotechar": '"', "fieldnames": ["id","text"]
                       })
# Document
# data = loader.load()
data = loader.lazy_load()
print(data)

for doc in data:
    print(doc)

