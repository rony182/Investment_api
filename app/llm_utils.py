import openai
import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

def query_llm(query, queried_data, max_tokens=300, temperature=0.5, top_p=0.95):
    try:
        # Extract narrative_texts from the queried data
        context = "\n".join([match['metadata']['narrative_texts'] for match in queried_data['matches'] if 'narrative_texts' in match['metadata']])
        
        if not context:
            logging.warning("No valid context found in Pinecone data.")
            return "I'm sorry, I couldn't find enough context to answer your query."

        prompt = (
            "Based on the following context, provide a detailed and expert-level response to the query. "
            "Ensure the response is well-structured, includes specific financial insights, comparisons to traditional financial instruments where relevant, and uses appropriate terminology.\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Query:\n"
            f"{query}\n\n"
            "Response:"
        )
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Or any other OpenAI model you wish to use
            messages=[
                {"role": "system", "content": "You are a finance and investment expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logging.error(f"Error querying LLM: {e}")
        return "There was an error processing your request. Please try again later."
