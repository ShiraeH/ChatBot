import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

def initialize_vectorstore():
    """
    Initialize Pinecone vector store. Create index if it does not exist.
    """
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX"]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Create index if not exists
    if index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=os.environ.get("PINECONE_REGION", "us-west-1"),
            ),
        )

    return LangchainPinecone.from_existing_index(index_name, embeddings)


if __name__ == "__main__":
    try:
        folder_path = "YOURPATH"
        file_paths = glob(os.path.join(folder_path, "*"), recursive=True)

        if not file_paths:
            print("No files found in the specified folder.")
            sys.exit()

        vectorstore = initialize_vectorstore()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        for file_path in file_paths:
            try:
                print(f"Processing file: {file_path}")
                loader = get_loader(file_path)

                if isinstance(loader, list) and "page_content" in loader[0]:
                    raw_docs = loader
                else:
                    raw_docs = loader.load()

                # `page_content` がないドキュメントをスキップし、適切な形式に変換
                formatted_docs = [
                    {"page_content": doc.get("page_content", ""), "metadata": doc.get("metadata", {})}
                    for doc in raw_docs if isinstance(doc, dict) and "page_content" in doc
                ]

                docs = text_splitter.split_documents(formatted_docs)
                vectorstore.add_documents(docs)
                print(f"Successfully processed: {file_path}")
            except Exception as file_error:
                print(f"Error processing {file_path}: {file_error}")
                print("Error! tmp stopp")
                break

        print("All files processed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
