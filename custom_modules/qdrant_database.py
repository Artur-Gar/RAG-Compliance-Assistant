import numpy as np
from tqdm.notebook import tqdm
from qdrant_client import QdrantClient, models
import sys


# Qdrant vector database
class PostEmbsQdrant:

    def __init__(self, 
                 embs_root, 
                 text_data, 
                 metadata_ids, 
                 collection_name, 
                 client_url='http://localhost:6333'): # localhost:6333 - запуск на локальном компьютере на стандартном порте 6333
        '''
        Initializes a connection to a Qdrant collection with prepared text and embeddings.

        Args:
        embs_root – path to the .npy file containing the embeddings  
        text_data – list of chunked text data  
        metadata_ids – list of IDs corresponding to the chunks  
        collection_name – name of the Qdrant collection (created if it doesn't exist)  
        client_url – URL of the Qdrant instance (e.g., 'http://localhost:6333' for local use)
        '''
        self.embs_root = embs_root
        self.data = [{'data': td, 'ids': md} for td, md in zip(text_data, metadata_ids)]
        self.collection_name = collection_name
        # Qdrant client setup (ensure Qdrant is running via Docker before this step)
        self.client = QdrantClient(url=client_url)

    def __call__(self):
        '''
        Loads precomputed embeddings from a .npy file and uploads them to a Qdrant collection.

        - Initializes the collection if it doesn't exist, setting vector size and storage preferences.
        - Resumes indexing from the current number of stored points if the collection already exists.
        - Converts embeddings into Qdrant point structures, assigning unique IDs and attaching metadata.
        - Splits the data into chunks (based on an estimated 25 MB limit) and uploads each chunk separately.
        '''
        embs = np.load(self.embs_root).tolist()

        if not self.client.collection_exists(self.collection_name):
            # Create the collection if it doesn't already exist 
            self.client.create_collection(
                collection_name=self.collection_name, #
                vectors_config=models.VectorParams(
                        size=len(embs[0]), # embedding sizes
                        distance=models.Distance.COSINE, 
                        on_disk=True # Store everything on disk instead of in memory 
                    ),                   
                on_disk_payload=True # Store metadata (payload) on disk instead of in memory 
            )

            points_count = 0 # The collection is empty as it was just initialized
        
        else:
            # For existing collections, fetch the current number of points to resume indexing correctly (points = records with unique integer IDs)
            counts = self.client.get_collection(self.collection_name).vectors_count # An existing but empty collection will return None
            points_count =  counts if counts else 0

        points=[
            models.PointStruct(
                id=idx + points_count,
                vector=vector, 
                payload=self.data[idx]
            )
            for idx, vector in enumerate(embs) # Each point represents an embedding vector of a text chunk
        ]

        # Calculate the size of the data in megabytes
        file_size_mb = sys.getsizeof(points) // 1024 + 1
        # Approximate how many records can fit in 25 MB (arbitrary value)
        chunk_size = int(25*(len(points) / file_size_mb))

        # Iterate through the chunks and add them to the database
        for i in tqdm(range(len(points) // chunk_size + 1)):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i*chunk_size: (i+1) * chunk_size] # Load all points from a specific chunk
            )

        print(f'Файлы успешно добавлены в базу, примерно {chunk_size} строк наблюдений влезет в 25 мегабайт')