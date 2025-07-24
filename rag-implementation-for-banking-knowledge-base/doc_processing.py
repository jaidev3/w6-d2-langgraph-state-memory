from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import tempfile
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#create a pinecone index if it doesn't exist
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Get list of existing indexes
existing_indexes = [index.name for index in pc.list_indexes()]

if "banking-knowledge-base" not in existing_indexes:
    pc.create_index(name="banking-knowledge-base", dimension=1536, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))



class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = PineconeVectorStore(index_name="banking-knowledge-base", embedding=self.embeddings)

    def process_pdf_files(self, pdf_files):
        all_docs = []
        for pdf_file in pdf_files:
            # Handle Streamlit UploadedFile objects
            if hasattr(pdf_file, 'read'):
                # This is a Streamlit UploadedFile object
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(pdf_file.read())
                    tmp_file_path = tmp_file.name
                
                try:
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    all_docs.extend(docs)
                finally:
                    # Clean up the temporary file
                    os.unlink(tmp_file_path)
            else:
                # This is a regular file path string
                loader = PyPDFLoader(pdf_file)
                docs = loader.load()
                all_docs.extend(docs)
        return all_docs

    def save_to_pinecone(self, chunks):
        self.vector_store.add_documents(chunks)

    def split_documents(self, docs):
        return self.text_splitter.split_documents(docs)

    def embed_documents(self, docs):
        return self.embeddings.embed_documents(docs)

# processor = DocumentProcessor()

# docs = processor.process_pdf_files(["rag-implementation-for-banking-knowledge-base/docs/banking_knowledge_base.pdf"])
# print(docs)

# chunks = processor.split_documents(docs)
# print(chunks)

# processor.save_to_pinecone(chunks)

print("Chunks saved to Pinecone")
