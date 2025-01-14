import os
import sys
from glob import glob
from docx import Document
from pptx import Presentation
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangchainPinecone
import pikepdf
import chardet
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import ssl
import certifi
ssl_context = ssl.create_default_context()
ssl_context.load_verify_locations(certifi.where())

load_dotenv()

def detect_file_encoding(file_path):
    """
    Detect the encoding of a file.
    """
    with open(file_path, "rb") as f:
        raw_data = f.read(1024)  # Read the first 1KB to detect encoding
    result = chardet.detect(raw_data)
    return result["encoding"]

def read_docx(file_path):
    """
    Read the content of a .docx file.
    """
    try:
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        raise ValueError(f"Error reading .docx file {file_path}: {e}")

def read_pdf(file_path):
    """
    Read the content of a PDF file using pikepdf.
    """
    try:
        with pikepdf.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        raise ValueError(f"Error reading PDF file {file_path}: {e}")

def validate_metadata(metadata):
    """
    Validate and correct metadata values for Pinecone.
    Converts unsupported types (e.g., dict) to strings.
    """
    valid_types = (str, int, float, bool, list)
    for key, value in metadata.items():
        if not isinstance(value, valid_types):
            if isinstance(value, dict):
                metadata[key] = str(value)
            else:
                raise ValueError(f"Unsupported metadata type for key '{key}': {type(value)}")
    return metadata

def initialize_vectorstore():
    """
    Initialize Pinecone vector store. Create index if it does not exist.
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    # Create index if it does not exist
    if index_name not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=os.getenv.get("PINECONE_REGION", "us-west-1")
            )
        )
    return LangchainPinecone.from_existing_index(index_name, embeddings)

def get_loader(file_path):
    """
    Return appropriate loader or document content based on file type.
    """
    if file_path.endswith(".pdf"):
        return read_pdf(file_path)
    elif file_path.endswith(".csv"):
        return CSVLoader(file_path)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        return UnstructuredExcelLoader(file_path)
    else:
        # Detect file encoding and use UnstructuredLoader
        encoding = detect_file_encoding(file_path)
        return UnstructuredLoader(file_path, encoding=encoding)

if __name__ == "__main__":
    try:
        folder_path = "C:/Users/cera0/Documents/Job/Proto241214/DATA/"
        file_paths = glob(os.path.join(folder_path, '**/*.*'), recursive=True)

        if not file_paths:
            print("No files found in the specified folder.")
            sys.exit()

        vectorstore = initialize_vectorstore()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        for file_path in file_paths:
            try:
                print(f"Processing file: {file_path}")
                loader = get_loader(file_path)

                # If the loader directly returns documents (e.g., for .docx)
                if isinstance(loader, str):  # For text content loaders
                    raw_docs = [{"page_content": loader}]
                else:  # For other loaders that require loading
                    raw_docs = loader.load()

                docs = text_splitter.split_documents(raw_docs)
                vectorstore.add_documents(docs)
                print(f"Successfully processed: {file_path}")
            except Exception as file_error:
                print(f"Error processing {file_path}: {file_error}")
                print("Error! tmp stopp")
                break

        print("All files processed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
