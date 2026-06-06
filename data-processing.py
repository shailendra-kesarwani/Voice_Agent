# import os

# import httpx
# from deepgram import (
#     DeepgramClient,
#     PrerecordedOptions,
#     FileSource,
# )
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from pytube import YouTube
from typing import List
from langchain_community.document_loaders import DirectoryLoader
from langchain_chroma import Chroma

load_dotenv(override=True)

def embed_text(text: str):
    # print("Embedding")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=256,
        length_function=len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits


def save_embeddings_to_db(docs: List[str]):
    embeddings = OpenAIEmbeddings()
    persist_directory = "./car-service-vectorestore"
    vectordb = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    # vectordb.persist()


if __name__ == "__main__":
    data_dir = '../car_service_data'
    loader = DirectoryLoader(data_dir, glob="*.md")
    docs = loader.load()
    embeded_docs = embed_text(docs)
    save_embeddings_to_db(docs=embeded_docs)