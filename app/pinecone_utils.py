# pinecone_utils.py

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from FlagEmbedding import FlagModel

# Load environment variables from the .env file
load_dotenv()

# Initialize Pinecone client
api_key = os.getenv("PINECONE_API_KEY")
environment = 'us-east-1'  # Your Pinecone environment region

pc = Pinecone(api_key=api_key)

# Connect to your specific index
index_name = "investments"
index = pc.Index(index_name)

# Initialize the embedding model using FlagEmbedding
model = FlagModel('BAAI/bge-small-en-v1.5', use_fp16=True)

def generate_embedding(query_text):
    try:
        query_embedding = model.encode([query_text])
        return query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def query_pinecone(query_embedding):
    try:
        response = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        if not response or 'matches' not in response or len(response['matches']) == 0:
            print("No matches found in Pinecone.")
            return None
        return response
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return None
