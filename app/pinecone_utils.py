import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from sklearn.decomposition import PCA
import numpy as np
import logging

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone client
api_key = os.getenv("PINECONE_API_KEY")
environment = 'us-east-1'  # Your Pinecone environment region

pc = Pinecone(api_key=api_key)

# Connect to your specific index
index_name = "investments"
index = pc.Index(index_name)

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Generate some sample data for PCA fitting
# Ideally, you should replace this with a representative sample of the data you'll be embedding
sample_data = np.random.randn(100, 1536)  # 100 random 1536-dimensional vectors

# Initialize and fit PCA for dimensionality reduction
pca = PCA(n_components=384)
pca.fit(sample_data)

def generate_embedding(query_text):
    try:
        logging.info("Generating embedding for query: %s", query_text)
        response = client.embeddings.create(
            input=query_text,
            model="text-embedding-ada-002"  # Produces 1536-dimensional embeddings
        )
        logging.info("Received response from OpenAI")

        # Extract the embedding vector from the response
        query_embedding = response.data[0].embedding

        # Convert to numpy array for PCA
        query_embedding = np.array(query_embedding).reshape(1, -1)
        logging.info("Converted embedding to numpy array")

        # Apply PCA to reduce to 384 dimensions
        reduced_embedding = pca.transform(query_embedding)
        logging.info("Applied PCA to reduce dimensions")

        return reduced_embedding.flatten().tolist()
    except Exception as e:
        logging.error(f"Error generating embedding with OpenAI: {e}")
        return None

def query_pinecone(query_embedding):
    try:
        response = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        if not response or 'matches' not in response or len(response['matches']) == 0:
            logging.warning("No matches found in Pinecone.")
            return None
        return response
    except Exception as e:
        logging.error(f"Error querying Pinecone: {e}")
        return None
