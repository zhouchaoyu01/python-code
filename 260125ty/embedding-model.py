from langchain_community.embeddings import DashScopeEmbeddings

embedding = DashScopeEmbeddings()  # Initialize the DashScopeEmbeddings model

embedding.embed_query("Hello, DashScope!")  # Get the embedding for the input text
embedding.embed_documents(["Hello, DashScope!", "Hello, LangChain!"])  # Get embeddings for multiple documents