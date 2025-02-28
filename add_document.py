import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone

def initialize_vectorstore():
    """
    Initialize Pinecone vector store. Create index if it does not exist.
    """
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    index_name = os.environ["PINECONE_INDEX"]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # インデックスが存在しない場合、作成する
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=3072,  # 変更しない場合は現在の次元数を確認
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=os.environ.get("PINECONE_REGION", "us-west-1"),
            ),
        )

    index = pc.Index(index_name)
    return LangchainPinecone(index, embeddings, text_key="page_content")
