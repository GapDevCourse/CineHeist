import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from datetime import datetime

# Load the input CSV file
input_file = r"C:\Users\KGRCET\Downloads\database.csv"
movies = pd.read_csv(input_file)
print(f"Loaded input file: {input_file} with {len(movies)} rows.")

# Automatically generate a new output file name
output_directory = os.path.dirname(input_file)  # Save in the same directory as the input file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate a timestamp
output_file = os.path.join(output_directory, f"database_with_vectors_{timestamp}.csv")

# Initialize the Sentence Transformer model
encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
print("Initialized the Sentence Transformer model.")

# Chunking function
def chunk_text(text, max_words=10):
    if pd.isna(text):  # Handle NaN entries
        return []
    words = text.split()
    chunks = [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
    return chunks[:50]  # Limit to the first 50 chunks to prevent overload

# Batch processing function
def encode_chunks_in_batches(chunks, model, batch_size=32):
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        embeddings.extend(model.encode(batch, show_progress_bar=False))
    return np.array(embeddings)

# Prepare columns for vectors and index
vectors_list = []
index_list = []

print("Starting vectorization process...")
for idx, row in movies.iterrows():
    if idx % 10 == 0:  # Log every 10 rows for visibility
        print(f"Processing row {idx+1}/{len(movies)}...")

    overview = row.get("overview", "")
    chunks = chunk_text(overview)

    if not chunks:
        vectors_list.append([])
        index_list.append(None)
        continue

    try:
        # Generate vectors for the chunks in batches
        vectors = encode_chunks_in_batches(chunks, encoder)
        vectors_list.append(vectors.tolist())  # Convert to a Python list for storage

        # Create and normalize the FAISS index
        vector_dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(vector_dimension)
        faiss.normalize_L2(vectors)
        index.add(vectors)

        # Save the index as binary for simplicity
        index_flat = index.reconstruct_n(0, index.ntotal)
        index_list.append(index_flat.tolist())  # Store index as list for CSV compatibility
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        vectors_list.append([])
        index_list.append(None)

# Add the vectors and FAISS index to the DataFrame
movies["vectors"] = vectors_list
movies["faiss_index"] = index_list

# Save the updated DataFrame to a new CSV file
try:
    movies.to_csv(output_file, index=False)
    print(f"Output saved to: {output_file}")
except Exception as e:
    print(f"Failed to save output file: {e}")
