from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize the SentenceTransformer model (use the same model as you used for movie descriptions)
encoder = SentenceTransformer("paraphrase-mpnet-base-v2")

# Function to encode the user prompt (e.g., a synopsis provided by the user)
def vectorize_prompt(prompt: str):
    # Encode the user prompt into a vector
    prompt_vector = encoder.encode([prompt])
    return prompt_vector

# Load the FAISS index (assuming it's stored in MongoDB or a file)
# Here, we'll assume you've saved it as a numpy array or other suitable format
def load_faiss_index(index_data):
    # Assuming index_data is a list of vectors you stored in MongoDB
    vectors = np.array(index_data).astype(np.float32)
    vector_dimension = vectors.shape[1]
    faiss_index = faiss.IndexFlatL2(vector_dimension)
    faiss_index.add(vectors)
    return faiss_index

# Function to perform similarity search using the FAISS index
def search_movies(prompt_vector, faiss_index, top_k=5):
    # Perform a similarity search on the FAISS index
    distances, indices = faiss_index.search(prompt_vector, top_k)
    return distances, indices

# Example usage

# Example user prompt (e.g., entered synopsis)
user_prompt = "A young couple struggles with their relationship while facing personal challenges."

# Vectorize the user prompt
prompt_vector = vectorize_prompt(user_prompt)

# Load the FAISS index (this is a mockup, you'll load the actual index data from your DB)
index_data_from_db = []  # Replace with actual data from MongoDB
faiss_index = load_faiss_index(index_data_from_db)

# Perform the search
distances, indices = search_movies(prompt_vector, faiss_index)

# Print out the results (similarity scores and corresponding movie indices)
print("Most similar movies:")
for i in range(len(indices[0])):
    print(f"Movie Index: {indices[0][i]}, Similarity Score: {distances[0][i]}")
