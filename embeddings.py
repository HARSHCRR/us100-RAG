from langchain_community.embeddings import FastEmbedEmbeddings

def get_embedding_model():
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# Test it
if __name__ == "__main__":
    model = get_embedding_model()
    
    test_text = "NQ was BULLISH after CPI beat"
    vector = model.embed_query(test_text)
    
    print(f"Model: BAAI/bge-small-en-v1.5")
    print(f"Vector dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
