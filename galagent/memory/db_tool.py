import pymongo
from pymongo import MongoClient, UpdateOne
from pymongo.errors import DuplicateKeyError, BulkWriteError
from bson.objectid import ObjectId

class PaperManager:
    def __init__(self, uri="mongodb://localhost:27017/", db_name="academic_db", collection_name="papers"):
        """
        初始化管理器
        :param uri: MongoDB 连接字符串
        :param maxPoolSize: 连接池最大连接数，高并发关键设置
        """
        # 1. 高并发配置：配置连接池
        # maxPoolSize: 允许的最大并发连接数 (默认100)，根据服务器负载调整
        # connectTimeoutMS: 连接超时时间
        self.client = MongoClient(
            uri, 
            maxPoolSize=200, 
            connectTimeoutMS=2000,
            serverSelectionTimeoutMS=3000
        )
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        
        # 2. 初始化索引 (只在启动时运行一次)
        self._init_indexes()

    def _init_indexes(self):
        """
        创建索引以支持高性能查询
        """
        # 针对 id 的唯一索引 (通常 _id 自带，但如果你使用自定义 id 字段)
        # self.collection.create_index("id", unique=True)
        
        # 核心需求：针对 keywords 的多键索引 (Multikey Index)
        # 这允许极快地查询 "keywords 包含 'data mining'" 的文章
        self.collection.create_index("keywords")
        
        # 针对 引用数 和 年份 的复合索引 (用于排序)
        self.collection.create_index([("n_citation", -1), ("year", -1)])
        
        # 针对 标题 的全文索引 (可选，用于模糊搜索)
        self.collection.create_index([("title", "text")])
        
        print("Indexes ensured.")

    # ==========================
    # C: Create (新增)
    # ==========================
    def add_paper(self, paper_data):
        """
        插入单篇论文
        """
        try:
            # 数据清洗：确保数字类型正确，方便排序
            if 'n_citation' in paper_data:
                paper_data['n_citation'] = int(paper_data['n_citation'])
            
            # 使用 insert_one
            result = self.collection.insert_one(paper_data)
            return str(result.inserted_id)
        except DuplicateKeyError:
            print(f"Paper with _id {paper_data.get('_id')} already exists.")
            return None

    def bulk_add_papers(self, papers_list):
        """
        批量插入 (高并发下推荐使用批量操作以减少网络IO)
        """
        if not papers_list:
            return
        try:
            # ordered=False 意味着其中一条失败不影响其他条目插入，提高并发吞吐量
            result = self.collection.insert_many(papers_list, ordered=False)
            print(f"Inserted {len(result.inserted_ids)} papers.")
        except BulkWriteError as bwe:
            print(f"Partial batch error: {bwe.details['writeErrors']}")

    # ==========================
    # R: Read (查询)
    # ==========================
    def search_by_keywords(self, keywords, limit=10, skip=0):
        """
        根据关键词查询，支持分页
        :param keywords: 列表 ["data mining", "social network"] (既包含A又包含B)
                         或者单个字符串 "data mining"
        """
        if isinstance(keywords, str):
            query = {"keywords": keywords}
        else:
            # $all 表示文章的关键词数组必须包含列表中的所有词
            query = {"keywords": {"$all": keywords}}

        # 只返回需要的字段，减少网络传输 (Projection)
        projection = {"title": 1, "authors.name": 1, "year": 1, "n_citation": 1, "abstract": 1}
        
        # 按引用数降序排列
        cursor = self.collection.find(query, projection)\
                                .sort("n_citation", -1)\
                                .skip(skip)\
                                .limit(limit)
        
        return list(cursor)

    def get_paper_detail(self, paper_id):
        """通过 ID 获取详情"""
        return self.collection.find_one({"_id": paper_id})

    # ==========================
    # U: Update (更新)
    # ==========================
    def atomic_increment_citation(self, paper_id, count=1):
        result = self.collection.update_one(
            {"_id": paper_id},
            {"$inc": {"n_citation": count}}  # 原子操作 $inc
        )
        return result.modified_count > 0

    def add_tag_to_paper(self, paper_id, new_keyword):
        self.collection.update_one(
            {"_id": paper_id},
            {"$addToSet": {"keywords": new_keyword}}
        )

    # ==========================
    # D: Delete (删除)
    # ==========================
    def delete_paper(self, paper_id):
        self.collection.delete_one({"_id": paper_id})

    def close(self):
        self.client.close()
    
    def get_papers_by_ids(self, id_list, fields=None):
        """
        批量根据 ID 列表获取论文数据。
        使用 MongoDB 的 $in 操作符，只需一次网络请求即可取回所有数据，速度极快。
        
        :param id_list: ID 列表，例如 ["id1", "id2", "id3"]
        :param fields: (可选) 指定返回字段，例如 {"title": 1, "year": 1}，默认返回全部
        :return: 包含结果文档的列表
        """
        if not id_list:
            return []

        # 1. 构建查询：使用 $in 操作符
        # MongoDB 会利用 _id 索引快速定位这一批文档
        query = {"_id": {"$in": id_list}}

        # 2. 执行查询
        # 如果调用者没有指定 fields，则使用默认的一个精简视图，或者你可以改为 None 返回所有字段
        if fields is None:
            # 默认只返回常用字段，减少网络传输负载
            fields = {"title": 1, "authors.name": 1, "year": 1, "n_citation": 1}

        cursor = self.collection.find(query, fields)
        
        # 3. 返回结果
        # 注意：MongoDB 返回的顺序不一定和 id_list 的顺序一致
        results = list(cursor)
        return results
    
    def get_references_by_paper_title(self, paper_title):
        """
        根据论文标题，查询它引用的所有参考文献详情。
        逻辑：
        1. 先根据 Title 找到源论文，提取 references 字段 (ID列表)。
        2. 再复用 get_papers_by_ids 批量获取这些 ID 的具体内容。
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
# 3. 使用示例 (模拟业务逻辑)
# ==========================
if __name__ == "__main__":
    manager = PaperManager()

    # --- 1. 构造第一条数据 (原有数据) ---
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
        "n_citation": 82, # 注意转为 int
        "doi": "10.4114/ia.v10i29.873",
        "abstract": "Our ability to generate..."
    }

    # --- 2. 构造第二条数据 (新增加的模拟数据) ---
    id_2 = "64f8a1b2c3d4e5f678901233" # 模拟一个新的 24位 hex ID
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
        "n_citation": 82, # 注意转为 int
        "doi": "10.4114/ia.v10i29.873",
        "abstract": "Our ability to generate..."
    }

    print("--- Inserting Data ---")
    # 插入数据 (如果已存在会提示重复，不影响后续查询)
    manager.add_paper(paper_1)
    manager.add_paper(paper_2)
    
    # --- 3. 测试批量 ID 查询接口 (核心测试点) ---
    print("\n--- Testing Batch Get by IDs ---")
    
    # 构造 ID 列表 (包含 id_1, id_2 和一个不存在的 ID)
    target_ids = [id_1, id_2, "不存在的ID_999"]
    
    print(f"Requesting IDs: {target_ids}")
    
    # 调用新接口
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