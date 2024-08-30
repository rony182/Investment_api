# llm_utils.py

from huggingface_hub import InferenceClient

# Initialize Hugging Face Inference Client
hf_client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

def query_llm(query, queried_data, max_tokens=300, temperature=0.5, top_p=0.95):
    try:
        # Extract narrative_texts from the queried data
        context = "\n".join([match['metadata']['narrative_texts'] for match in queried_data['matches'] if 'narrative_texts' in match['metadata']])
        
        if not context:
            print("No valid context found in Pinecone data.")
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
        
        messages = [
            {"role": "system", "content": "You are a finance and investment expert."},
            {"role": "user", "content": prompt}
        ]
        
        response = ""
        for message in hf_client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p
        ):
            token = message.choices[0].delta.content
            response += token

        return response.strip()
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return "There was an error processing your request. Please try again later."
