import pandas as pd 
from sentence_transformers import SentenceTransformer
import faiss

movies = pd.read_csv("tmdb dataset/database.csv")

def chunk_text(text, max_words = 10):
    words = text.split(" ")
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks


chunk_map = {}

for idx, row in movies.iterrows():
    overview = row["overview"]
    chunks = chunk_text(overview)
    print(overview)
    print(chunks)
    print(len(chunks))

    for chunk in chunks:
        chunk_map[chunk] = idx 
        
    break
encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
vectors = encoder.encode(chunks)
all_vectors=[]
all_vectors.append(vectors)
print(all_vectors)

vector_dimension = vectors.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(vectors)
index.add(vectors)
print(index)