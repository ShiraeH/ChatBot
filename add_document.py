import os
import sys
import pinecone
from glob import glob
from pathlib import Path
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangchainPinecone

load_dotenv()

def initialize_vectorstore():
    """
    Initialize Pinecone vector store. Create index if it does not exist.
    """
    pinecone.init(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX"]
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )
    # Create index if it does not exist
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=3072,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=os.environ.get("PINECONE_REGION", "us-west-1")
            )
        )

    index = pinecone.Index(index_name)
    return LangchainPinecone(index, embeddings, text_key="page_content")
