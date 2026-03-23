import pymongo
from pymongo import MongoClient, UpdateOne
from pymongo.errors import DuplicateKeyError, BulkWriteError
from bson.objectid import ObjectId

class PaperManager:
    def __init__(self, uri="mongodb://localhost:27017/", db_name="academic_db", collection_name="papers"):
        """
        Initialize the manager
        :param uri: MongoDB connection string
        :param maxPoolSize: Maximum number of connections in the pool (key setting for high concurrency)
        """
        # 1. High-concurrency configuration: configure connection pool
        # maxPoolSize: maximum number of concurrent connections allowed (default 100); adjust based on server load
        # connectTimeoutMS: connection timeout
        self.client = MongoClient(
            uri, 
            maxPoolSize=200, 
            connectTimeoutMS=2000,
            serverSelectionTimeoutMS=3000
        )
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        
        # 2. Initialize indexes (runs only once at startup)
        self._init_indexes()

    def _init_indexes(self):
        """
        Create indexes to support high-performance queries
        """
        # Unique index on id (usually _id has it, but if you use a custom id field)
        # self.collection.create_index("id", unique=True)

        # Core requirement: multikey index on keywords
        # This allows extremely fast queries for articles "whose keywords contain 'data mining'"
        self.collection.create_index("keywords")
        
        # Compound index on citation count and year (for sorting)
        self.collection.create_index([("n_citation", -1), ("year", -1)])

        # Full-text index on title (optional, for fuzzy search)
        self.collection.create_index([("title", "text")])
        
        print("Indexes ensured.")

    # ==========================
    # C: Create
    # ==========================
    def add_paper(self, paper_data):
        """
        Insert a single paper
        """
        try:
            # Data cleaning: ensure numeric types are correct for sorting
            if 'n_citation' in paper_data:
                paper_data['n_citation'] = int(paper_data['n_citation'])
            
            # Use insert_one
            result = self.collection.insert_one(paper_data)
            return str(result.inserted_id)
        except DuplicateKeyError:
            print(f"Paper with _id {paper_data.get('_id')} already exists.")
            return None

    def bulk_add_papers(self, papers_list):
        """
        Batch insert (recommended under high concurrency to reduce network IO)
        """
        if not papers_list:
            return
        try:
            # ordered=False means a failure on one record does not block others, improving concurrent throughput
            result = self.collection.insert_many(papers_list, ordered=False)
            print(f"Inserted {len(result.inserted_ids)} papers.")
        except BulkWriteError as bwe:
            print(f"Partial batch error: {bwe.details['writeErrors']}")

    # ==========================
    # R: Read
    # ==========================
    def search_by_keywords(self, keywords, limit=10, skip=0):
        """
        Query by keywords, supports pagination
        :param keywords: List ["data mining", "social network"] (must contain both A and B)
                         or a single string "data mining"
        """
        if isinstance(keywords, str):
            query = {"keywords": keywords}
        else:
            # $all means the article's keyword array must contain all words in the list
            query = {"keywords": {"$all": keywords}}

        # Only return required fields, reduce network transfer (Projection)
        projection = {"title": 1, "authors.name": 1, "year": 1, "n_citation": 1, "abstract": 1}

        # Sort by citation count descending
        cursor = self.collection.find(query, projection)\
                                .sort("n_citation", -1)\
                                .skip(skip)\
                                .limit(limit)
        
        return list(cursor)

    def get_paper_detail(self, paper_id):
        """Get details by ID"""
        return self.collection.find_one({"_id": paper_id})

    # ==========================
    # U: Update
    # ==========================
    def atomic_increment_citation(self, paper_id, count=1):
        result = self.collection.update_one(
            {"_id": paper_id},
            {"$inc": {"n_citation": count}}  # Atomic operation $inc
        )
        return result.modified_count > 0

    def add_tag_to_paper(self, paper_id, new_keyword):
        self.collection.update_one(
            {"_id": paper_id},
            {"$addToSet": {"keywords": new_keyword}}
        )

    # ==========================
    # D: Delete
    # ==========================
    def delete_paper(self, paper_id):
        self.collection.delete_one({"_id": paper_id})

    def close(self):
        self.client.close()
    
    def get_papers_by_ids(self, id_list, fields=None):
        """
        Batch-fetch paper data by ID list.
        Uses MongoDB's $in operator, retrieving all data in a single network request for maximum speed.

        :param id_list: List of IDs, e.g. ["id1", "id2", "id3"]
        :param fields: (Optional) Specify returned fields, e.g. {"title": 1, "year": 1}; defaults to returning all
        :return: List of result documents
        """
        if not id_list:
            return []

        # 1. Build query: use the $in operator
        # MongoDB uses the _id index to quickly locate this batch of documents
        query = {"_id": {"$in": id_list}}

        # 2. Execute query
        # If the caller does not specify fields, use a default concise view, or change to None to return all fields
        if fields is None:
            # Default to only returning commonly used fields to reduce network transfer load
            fields = {"title": 1, "authors.name": 1, "year": 1, "n_citation": 1}

        cursor = self.collection.find(query, fields)

        # 3. Return results
        # Note: MongoDB does not guarantee the returned order matches id_list order
        results = list(cursor)
        return results
    
    def get_references_by_paper_title(self, paper_title):
        """
        Given a paper title, query the details of all references it cites.
        Logic:
        1. First find the source paper by title and extract the references field (list of IDs).
        2. Then reuse get_papers_by_ids to batch-fetch the content of those IDs.
        """
        source_paper = self.collection.find_one(
            {"title": paper_title}, 
            {"references": 1}
        )
        if not source_paper:
            print(f"Warning: Paper '{paper_title}' not found.")
            return []
        ref_ids = source_paper.get("references", [])
        if not ref_ids:
            print(f"Paper '{paper_title}' has no references.")
            return []

        return self.get_papers_by_ids(ref_ids)

# ==========================
# 3. Usage example (simulated business logic)
# ==========================
if __name__ == "__main__":
    manager = PaperManager()

    # --- 1. Construct the first record (original data) ---
    id_1 = "53e9ab9eb7602d970354a97i"
    paper_1 = {
        "_id": "53e9ab9eb7602d970354a97i", 
        "title": "Data mining: concepts and techniques test 2",
        "authors": [
            {"id": "53f42f36dabfaedce54dcd0c", "name": "Jiawei Han", "org": "UIUC", "org_id": 157725225}
        ],
        "venue": {"id": "53e17f5b20f7dfbc07e8ac6e", "name": "Inteligencia Artificial"},
        "year": 2000,
        "keywords": ["data mining", "structured data", "world wide web"],
        "references": ["53e9ab9eb7602d970354a97g", "64f8a1b2c3d4e5f678901233"],
        "n_citation": 82, # Note: convert to int
        "doi": "10.4114/ia.v10i29.873",
        "abstract": "Our ability to generate..."
    }

    # --- 2. Construct the second record (newly added simulated data) ---
    id_2 = "64f8a1b2c3d4e5f678901233" # Simulated new 24-character hex ID
    paper_2 = {
        "_id": "64f8a1b2c3d4e5f678901233",
        "title": "Data mining: concepts and techniques_1",
        "authors": [
            {"id": "53f42f36dabfaedce54dcd0c", "name": "Jiawei Han", "org": "UIUC", "org_id": 157725225}
        ],
        "venue": {"id": "53e17f5b20f7dfbc07e8ac6e", "name": "Inteligencia Artificial"},
        "year": 2000,
        "keywords": ["data mining", "structured data", "world wide web"],
        "references": ["53e9ab9eb7602d970354a97e", "53e9ab9eb7602d970354a97e"],
        "n_citation": 82, # Note: convert to int
        "doi": "10.4114/ia.v10i29.873",
        "abstract": "Our ability to generate..."
    }

    print("--- Inserting Data ---")
    # Insert data (duplicate entries will be reported but won't affect subsequent queries)
    manager.add_paper(paper_1)
    manager.add_paper(paper_2)

    # --- 3. Test the batch ID query interface (core test point) ---
    print("\n--- Testing Batch Get by IDs ---")

    # Build the ID list (containing id_1, id_2, and a non-existent ID)
    target_ids = [id_1, id_2, "non_existent_ID_999"]

    print(f"Requesting IDs: {target_ids}")

    # Call the new interface
    batch_results = manager.get_papers_by_ids(target_ids)
    
    print(f"Found {len(batch_results)} documents:\n")
    for p in batch_results:
        print(f"✅ Found: [{p.get('_id')}]")
        print(f"   Title: {p.get('title')}")
        print(f"   Year : {p.get('year')}")
        print("-" * 30)

    reference_test = manager.get_references_by_paper_title("Data mining: concepts and techniques test 2")
    print(reference_test)
    manager.close()