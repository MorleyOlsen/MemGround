# Faiss database management layer
import faiss
import numpy as np
import threading
import os
import pickle
import json

class ThreadSafeFaissManager:
    def __init__(self, dim, index_file=None, id_map_file=None, metric='L2'):
        """
        Initialize the FAISS manager
        :param dim: Vector dimension
        :param index_file: Path to the index file (for loading)
        :param id_map_file: Path to the ID mapping file (if additional metadata needs to be managed)
        :param metric: 'L2' (Euclidean distance) or 'IP' (inner product / cosine similarity)
        """
        self.dim = dim
        self.lock = threading.Lock()  # Core: thread lock ensures all operations that mutate the index state are protected
        #self.documents = []
        #if not id_map_file == None:
        #    self.documents = self.read_documents(id_map_file)
        
        if id_map_file and os.path.exists(id_map_file):
            try:
                with open(id_map_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # !!! Critical change: convert to numpy array to support batch indexing !!!
                    self.id_map = np.array(data) 
            except Exception as e:
                print(f"Warning: failed to load mapping file: {e}")
                self.id_map = np.array([]) # Keep type consistent
        else:
            self.id_map = np.array([])
            
        self.documents = id_map_file
        # Determine the metric type
        self.metric = faiss.METRIC_L2 if metric == 'L2' else faiss.METRIC_INNER_PRODUCT

        # Load or create a new index
        if index_file and os.path.exists(index_file):
            print(f"Loading index: {index_file}")
            self.index = faiss.read_index(index_file)
        else:
            print("Initializing new index (IndexIDMap + IndexFlatL2/IP)...")
            # Use IndexFlat as the base index
            if metric == 'L2':
                base_index = faiss.IndexFlatL2(dim)
            else:
                base_index = faiss.IndexFlatIP(dim)
            
            # Wrap with IndexIDMap to support add_with_ids
            self.index = faiss.IndexIDMap(base_index)
        
    def read_documents(id_map_file):
        import glob
        file_list = []
        documents = glob.glob(id_map_file)
        for file_path in documents:
            with open(file_path , "r" , encoding = "utf-8") as f:
                doc = f.read()
                file_list.append(doc)
        return file_list
    
    def _to_float32(self, vectors):
        """Helper function: ensure data is in float32 format"""
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
        if vectors.dtype != 'float32':
            vectors = vectors.astype('float32')
        return vectors

    def add_vectors(self, vectors, ids):
        """
        Thread-safe vector addition
        :param vectors: List of vectors or numpy array (shape: [n, dim])
        :param ids: Corresponding list of IDs (shape: [n])
        """
        vectors = self._to_float32(vectors)
        ids = np.array(ids, dtype=np.int64)

        if len(vectors) != len(ids):
            raise ValueError("Number of vectors does not match number of IDs")

        # Acquire lock: FAISS add operations are not thread-safe
        try:
            with self.lock:
                self.index.add_with_ids(vectors, ids)
        except:
            return False
        return True
        # In a real scenario, this is where you would typically update the database or metadata cache

    def search(self, query_vectors, k=5):
        query_vectors = self._to_float32(query_vectors)
        try:
            with self.lock:
                distances, indices = self.index.search(query_vectors, k)
                
                if len(self.id_map) == 0:
                    return distances, indices
                mapped_ids = np.empty(indices.shape, dtype=object)
                valid_mask = indices != -1
                safe_mask = valid_mask & (indices < len(self.id_map))
                mapped_ids[safe_mask] = self.id_map[indices[safe_mask]]
                mapped_ids[~safe_mask] = None 
                return distances, mapped_ids

        except Exception as e:
            print(f"Search error: {e}")
            return None, None

    def remove_ids(self, ids):
        """Thread-safe vector deletion"""
        ids = np.array(ids, dtype=np.int64)
        try:
            with self.lock:
                self.index.remove_ids(ids)
        except:
            return False
    def count(self):
        """Get the number of vectors currently in the index"""
        return self.index.ntotal

    def save(self, index_file):
        """Save the index to disk"""
        with self.lock:
            print(f"Saving index to {index_file}...")
            faiss.write_index(self.index, index_file)
            print("Save complete.")
        return True

import numpy as np
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

class FaissDB:
    def __init__(self, dim, db_path, id_map_dir = "./paper_ids" , num_shards=32, metric='L2'):
        """
        Initialize a sharded FAISS database
        :param dim: Vector dimension
        :param db_path: Main directory for database storage
        :param num_shards: Number of shards (default 32)
        :param metric: 'L2' or 'IP'
        """
        self.dim = dim
        self.num_shards = num_shards
        self.db_path = db_path
        self.metric = metric
        self.shards = []

        # Ensure the directory exists
        if not os.path.exists(db_path):
            os.makedirs(db_path)

        if not os.path.exists(id_map_dir):
            os.makedirs(id_map_dir)

        print(f"Initializing FaissDB (num_shards: {num_shards})...")
        print(f"Mapping file directory: {id_map_dir}")

        for i in range(num_shards):
            # 1. Index file path
            index_file = os.path.join(db_path, f"shard_{i}.index")
            
            # 2. [Modified logic] Build the corresponding ID mapping file path
            # Format: ./paper_ids/id_map_0.json
            map_file = os.path.join(id_map_dir, f"id_map_{i}.json")
            
            # 3. Pass to Manager
            manager = ThreadSafeFaissManager(
                dim, 
                index_file=index_file, 
                id_map_file=map_file, 
                metric=metric
            )
            self.shards.append(manager)

    def _get_shard_id(self, vector_id):
        """Routing strategy: determine the shard based on ID"""
        return vector_id % self.num_shards

    def add_vectors(self, vectors, ids):
        """
        Dispatch and add: route vectors to their corresponding shard based on ID
        """
        vectors = np.array(vectors).astype('float32')
        ids = np.array(ids).astype('int64')
        
        if len(vectors) != len(ids):
            raise ValueError("Vectors and IDs length mismatch")

        # 1. Bin data by shard
        # shard_data = { shard_id: ([vectors], [ids]) }
        shard_data = {i: ([], []) for i in range(self.num_shards)}
        
        for vec, vid in zip(vectors, ids):
            shard_id = self._get_shard_id(vid)
            shard_data[shard_id][0].append(vec)
            shard_data[shard_id][1].append(vid)

        # 2. Batch write to each shard
        success_count = 0
        for shard_id, (vecs, vids) in shard_data.items():
            if vecs:
                # Call the underlying Manager's add method
                result = self.shards[shard_id].add_vectors(np.array(vecs), np.array(vids))
                if result:
                    success_count += len(vecs)
        
        return success_count

    def search(self, query_vectors, k=5):
        """
        Search all shards in parallel and merge results (Map-Reduce)
        """
        query_vectors = np.array(query_vectors).astype('float32')
        num_queries = len(query_vectors)
        
        # Result containers
        all_distances = []
        all_indices = []

        # 1. Parallel search (Scatter)
        # Use a thread pool because FAISS releases the GIL internally; multi-threading suits IO/C++ compute-intensive work
        with ThreadPoolExecutor(max_workers=self.num_shards) as executor:
            future_to_shard = {
                executor.submit(shard.search, query_vectors, k): i 
                for i, shard in enumerate(self.shards)
            }
            
            # Collect results
            results = [None] * self.num_shards
            for future in as_completed(future_to_shard):
                shard_idx = future_to_shard[future]
                dists, idxs = future.result()
                if dists is not None:
                    results[shard_idx] = (dists, idxs)
                else:
                    # If a shard search fails or is empty, fill with maximum distance / invalid ID
                    # Use float('inf') for L2, or -float('inf') for IP
                    fail_dist = float('inf') if self.metric == 'L2' else -float('inf')
                    results[shard_idx] = (
                        np.full((num_queries, k), fail_dist, dtype='float32'),
                        np.full((num_queries, k), -1, dtype='int64')
                    )

        # 2. Merge results (Gather)
        # results is now 32 tuples, each containing (N, k) distances and indices
        # We need to concatenate them into (N, 32*k)

        # Extract all dists and idxs
        batch_dists = [r[0] for r in results] # List of 32 (N, k) arrays
        batch_idxs = [r[1] for r in results]  # List of 32 (N, k) arrays
        
        # Concatenate along axis=1 -> (N, 32*k)
        merged_dists = np.concatenate(batch_dists, axis=1)
        merged_idxs = np.concatenate(batch_idxs, axis=1)
        
        # 3. Sort and take Top K (Reduce)
        final_dists = []
        final_idxs = []
        
        for i in range(num_queries):
            # Get all results for a single query vector
            d_row = merged_dists[i]
            i_row = merged_idxs[i]
            
            # Sort based on metric
            if self.metric == 'L2':
                # L2: smaller distance is better -> ascending order
                sorted_indices = np.argsort(d_row)
            else:
                # IP: larger inner product is better -> descending (argsort is ascending by default, so reverse)
                sorted_indices = np.argsort(d_row)[::-1]

            # Take the top k
            top_k_indices = sorted_indices[:k]
            
            final_dists.append(d_row[top_k_indices])
            final_idxs.append(i_row[top_k_indices])
            
        return np.array(final_dists), np.array(final_idxs)

    def remove_ids(self, ids):
        """
        Route deletion: calculate which shard the ID belongs to and delete
        """
        ids = np.array(ids, dtype=np.int64)
        shard_map = {i: [] for i in range(self.num_shards)}
        
        for vid in ids:
            shard_id = self._get_shard_id(vid)
            shard_map[shard_id].append(vid)
            
        for shard_id, vids in shard_map.items():
            if vids:
                self.shards[shard_id].remove_ids(vids)

    def save(self):
        """Save all shards"""
        print("Saving all shards...")
        for i, shard in enumerate(self.shards):
            # The index_file was specified in __init__ via ThreadSafeFaissManager
            # We need to ensure ThreadSafeFaissManager remembers the path, or we pass it again
            # The original class save requires a filename, so we rebuild the path here
            path = os.path.join(self.db_path, f"shard_{i}.index")
            shard.save(path)
        print("All shards saved.")

    def count(self):
        """Count total number of vectors"""
        total = 0
        for shard in self.shards:
            total += shard.count()
        return total