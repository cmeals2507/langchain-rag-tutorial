from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import os
import shutil

from langchain.embeddings import OpenAIEmbeddings

from langchain.embeddings import OpenAIEmbeddings

openai_ef = OpenAIEmbeddings(
                api_key="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",
                model_name="text-embedding-3-small"
            )


CHROMA_PATH = "/Users/cdmeals/opt/anaconda3/lib/python3.9/site-packages/chromadb"
DATA_PATH = "/Users/cdmeals/Library/CloudStorage/Dropbox/UH 2023.2024/Research AY 2023-2024/AI Rubric/Code/data"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_faiss(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=30,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_faiss(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = FAISS.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
