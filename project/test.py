import pymongo
import numpy as np
from bson.binary import Binary
import faiss

# Function to connect to MongoDB and retrieve the vector data
def get_vectors_from_mongodb(mongo_uri, db_name, collection_name):
    # Connect to MongoDB
    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Retrieve all documents (you can filter or limit the data as needed)
    cursor = collection.find({})  # Fetch all documents (customize if needed)

    vectors = []
    for doc in cursor:
        # Assuming each document contains the 'vectors' field which stores the vectors
        vector_data = doc.get('vectors')  # This should be a list of vectors
        
        # If vectors are stored as Binary or in another format, convert them
        if isinstance(vector_data, Binary):
            vector_data = np.frombuffer(vector_data, dtype=np.float32)
        
        # Ensure the vector has a valid shape and append to the list
        if vector_data is not None and len(vector_data) > 0:
            # Ensure all vectors are of the same dimension (e.g., 5)
            vectors.append(vector_data)
        else:
            print(f"Skipping empty or invalid vector for movie ID: {doc.get('_id')}")
    
    # Close the MongoDB connection
    client.close()

    # Now, validate all vectors are the same length and filter out those that are not
    if vectors:
        vector_length = len(vectors[0])  # Get the length of the first vector
        # Filter out vectors that have different lengths
        filtered_vectors = [vec for vec in vectors if len(vec) == vector_length]
    else:
        filtered_vectors = []

    # Convert list of vectors to a NumPy array
    if filtered_vectors:
        try:
            vectors_array = np.array(filtered_vectors)
        except Exception as e:
            print(f"Error converting vectors to NumPy array: {e}")
            raise
    else:
        raise ValueError("No valid vectors found or all vectors have different dimensions.")

    return vectors_array

# Example usage
mongo_uri = "mongodb+srv://saisabarishwins:Sabarish18@cineheist.6om63.mongodb.net/"  # Your MongoDB URI
db_name = "CineHeist"           # Replace with your database name
collection_name = "Movies_with_vectors" # Replace with your collection name

try:
    # Retrieve vectors from MongoDB
    vectors = get_vectors_from_mongodb(mongo_uri, db_name, collection_name)

    # Now you have the vectors as a NumPy array, and you can use them in FAISS
    print(f"Retrieved {vectors.shape[0]} vectors with dimension {vectors.shape[1]}")

    # Example: Creating a FAISS index from the vectors
    vector_dimension = vectors.shape[1]
    faiss_index = faiss.IndexFlatL2(vector_dimension)

    # Adding the vectors to the FAISS index
    faiss_index.add(vectors.astype(np.float32))  # Ensure the vectors are float32 type

    # Now you can perform similarity search using this FAISS index
    print("FAISS index created and vectors added successfully.")

except Exception as e:
    print(f"Error: {e}")
