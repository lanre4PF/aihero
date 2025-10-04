from sentence_transformers import SentenceTransformer
import numpy as np 
from tqdm import tqdm

def generate_embeddings(chunks):
    embeddings = []
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    
    for d in tqdm(chunks, desc="Generating embeddings"):
        # print(d)  # Add this line to inspect the structure
        text = d['content']
        embedding = model.encode(text)
        embeddings.append(embedding)
    embeddings = np.array(embeddings)
    print(f"embeddings shape: {embeddings.shape}")

    return embeddings

# if __name__ == "__main__":
#     generate_embeddings()
