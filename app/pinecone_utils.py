import os
from dotenv import load_dotenv
from pinecone import Pinecone
import openai
from sklearn.decomposition import PCA
import numpy as np  # Import NumPy
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
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize PCA for dimensionality reduction
pca = PCA(n_components=384)

def generate_embedding(query_text):
    try:
        logging.info("Generating embedding for query: %s", query_text)
        response = openai.embeddings_utils.get_embedding(
            input=query_text,
            model="text-embedding-ada-002"  # Produces 1536-dimensional embeddings
        )
        logging.info("Received response from OpenAI")

        # Extract the embedding vector from the response
        query_embedding = response

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
