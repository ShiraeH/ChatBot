import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone

# .envファイルを読み込む
load_dotenv()

def initialize_vectorstore():
    """
    Initialize Pinecone vector store. Create index if it does not exist.
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index_name = os.getenv("PINECONE_INDEX")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # インデックスが存在しない場合に作成
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=os.getenv("PINECONE_REGION", "us-west-1"),
            ),
        )

    index = pc.Index(index_name)
    return LangchainPinecone(index, embeddings, text_key="page_content")
