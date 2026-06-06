from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb
from dotenv import load_dotenv
load_dotenv(override=True)

documents = SimpleDirectoryReader("../car_service_data/").load_data()
embedding_model = OpenAIEmbedding(model="text-embedding-3-small")

db = chromadb.PersistentClient(path="./car-service-vectorestore-llama-index")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embedding_model
)