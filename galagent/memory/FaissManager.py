# Faiss数据库管理层
import faiss
import numpy as np
import threading
import os
import pickle
import json

class ThreadSafeFaissManager:
    def __init__(self, dim, index_file=None, id_map_file=None, metric='L2'):
        """
        初始化 FAISS 管理器
        :param dim: 向量维度
        :param index_file: 索引文件路径（用于加载）
        :param id_map_file: ID映射文件路径（如有额外元数据需管理）
        :param metric: 'L2' (欧氏距离) or 'IP' (内积/余弦相似度)
        """
        self.dim = dim
        self.lock = threading.Lock()  # 核心：线程锁确保所有会改动索引状态的操作都被锁保护
        #self.documents = []
        #if not id_map_file == None:
        #    self.documents = self.read_documents(id_map_file)
        
        if id_map_file and os.path.exists(id_map_file):
            try:
                with open(id_map_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # !!! 关键修改：转为 numpy array 以支持批量索引 !!!
                    self.id_map = np.array(data) 
            except Exception as e:
                print(f"⚠️ 加载映射文件失败: {e}")
                self.id_map = np.array([]) # 保持类型一致
        else:
            self.id_map = np.array([])
            
        self.documents = id_map_file
        # 确定度量方式
        self.metric = faiss.METRIC_L2 if metric == 'L2' else faiss.METRIC_INNER_PRODUCT
        
        # 加载或新建索引
        if index_file and os.path.exists(index_file):
            print(f"正在加载索引: {index_file}")
            self.index = faiss.read_index(index_file)
        else:
            print("初始化新索引 (IndexIDMap + IndexFlatL2/IP)...")
            # 使用 IndexFlat 作为基础索引
            if metric == 'L2':
                base_index = faiss.IndexFlatL2(dim)
            else:
                base_index = faiss.IndexFlatIP(dim)
            
            # 使用 IndexIDMap 包装，以便支持 add_with_ids
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
        """辅助函数：确保数据是 float32 格式"""
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
        if vectors.dtype != 'float32':
            vectors = vectors.astype('float32')
        return vectors

    def add_vectors(self, vectors, ids):
        """
        线程安全地添加向量
        :param vectors: 向量列表或 numpy array (shape: [n, dim])
        :param ids: 对应的 ID 列表 (shape: [n])
        """
        vectors = self._to_float32(vectors)
        ids = np.array(ids, dtype=np.int64)

        if len(vectors) != len(ids):
            raise ValueError("向量数量与ID数量不匹配")

        # 加锁：FAISS 的 add 操作不是线程安全的
        try:
            with self.lock:
                self.index.add_with_ids(vectors, ids)
        except:
            return False
        return True
        # 实际场景中，这里通常还会更新数据库或元数据缓存

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
        """线程安全地删除向量"""
        ids = np.array(ids, dtype=np.int64)
        try:
            with self.lock:
                self.index.remove_ids(ids)
        except:
            return False
    def count(self):
        """获取当前索引中的向量数量"""
        return self.index.ntotal

    def save(self, index_file):
        """保存索引到磁盘"""
        with self.lock:
            print(f"正在保存索引到 {index_file}...")
            faiss.write_index(self.index, index_file)
            print("保存完成。")
        return True

import numpy as np
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

class FaissDB:
    def __init__(self, dim, db_path, id_map_dir = "./paper_ids" , num_shards=32, metric='L2'):
        """
        初始化分片式 FAISS 数据库
        :param dim: 向量维度
        :param db_path: 数据库存储的主目录
        :param num_shards: 分片数量 (默认32)
        :param metric: 'L2' or 'IP'
        """
        self.dim = dim
        self.num_shards = num_shards
        self.db_path = db_path
        self.metric = metric
        self.shards = []

        # 确保目录存在
        if not os.path.exists(db_path):
            os.makedirs(db_path)

        if not os.path.exists(id_map_dir):
            os.makedirs(id_map_dir)

        print(f"正在初始化 FaissDB (分片数: {num_shards})...")
        print(f"映射文件目录: {id_map_dir}")

        for i in range(num_shards):
            # 1. 索引文件路径
            index_file = os.path.join(db_path, f"shard_{i}.index")
            
            # 2. [修改逻辑] 构造对应的 ID 映射文件路径
            # 格式: ./paper_ids/id_map_0.json
            map_file = os.path.join(id_map_dir, f"id_map_{i}.json")
            
            # 3. 传递给 Manager
            manager = ThreadSafeFaissManager(
                dim, 
                index_file=index_file, 
                id_map_file=map_file, 
                metric=metric
            )
            self.shards.append(manager)

    def _get_shard_id(self, vector_id):
        """路由策略：根据 ID 决定分片"""
        return vector_id % self.num_shards

    def add_vectors(self, vectors, ids):
        """
        分发添加：将向量根据 ID 路由到对应的分片中
        """
        vectors = np.array(vectors).astype('float32')
        ids = np.array(ids).astype('int64')
        
        if len(vectors) != len(ids):
            raise ValueError("Vectors and IDs length mismatch")

        # 1. 将数据按分片归类 (Binning)
        # shard_data = { shard_id: ([vectors], [ids]) }
        shard_data = {i: ([], []) for i in range(self.num_shards)}
        
        for vec, vid in zip(vectors, ids):
            shard_id = self._get_shard_id(vid)
            shard_data[shard_id][0].append(vec)
            shard_data[shard_id][1].append(vid)

        # 2. 批量写入各个分片
        success_count = 0
        for shard_id, (vecs, vids) in shard_data.items():
            if vecs:
                # 调用底层 Manager 的添加方法
                result = self.shards[shard_id].add_vectors(np.array(vecs), np.array(vids))
                if result:
                    success_count += len(vecs)
        
        return success_count

    def search(self, query_vectors, k=5):
        """
        并行搜索所有分片并合并结果 (Map-Reduce)
        """
        query_vectors = np.array(query_vectors).astype('float32')
        num_queries = len(query_vectors)
        
        # 结果容器
        all_distances = []
        all_indices = []

        # 1. 并行搜索 (Scatter)
        # 使用线程池，因为 FAISS 底层会释放 GIL，IO/C++计算密集型适合多线程
        with ThreadPoolExecutor(max_workers=self.num_shards) as executor:
            future_to_shard = {
                executor.submit(shard.search, query_vectors, k): i 
                for i, shard in enumerate(self.shards)
            }
            
            # 收集结果
            results = [None] * self.num_shards
            for future in as_completed(future_to_shard):
                shard_idx = future_to_shard[future]
                dists, idxs = future.result()
                if dists is not None:
                    results[shard_idx] = (dists, idxs)
                else:
                    # 如果某个分片搜索失败或为空，填充最大距离/无效ID
                    # 使用 float('inf') 对于 L2，或者 -float('inf') 对于 IP
                    fail_dist = float('inf') if self.metric == 'L2' else -float('inf')
                    results[shard_idx] = (
                        np.full((num_queries, k), fail_dist, dtype='float32'),
                        np.full((num_queries, k), -1, dtype='int64')
                    )

        # 2. 合并结果 (Gather)
        # 现在的 results 是 32 个 tuple，每个包含 (N, k) 的 distances 和 indices
        # 我们需要将其拼接成 (N, 32*k)
        
        # 提取所有的 dists 和 idxs
        batch_dists = [r[0] for r in results] # List of 32 (N, k) arrays
        batch_idxs = [r[1] for r in results]  # List of 32 (N, k) arrays
        
        # 沿 axis=1 拼接 -> (N, 32*k)
        merged_dists = np.concatenate(batch_dists, axis=1)
        merged_idxs = np.concatenate(batch_idxs, axis=1)
        
        # 3. 排序截取 Top K (Reduce)
        final_dists = []
        final_idxs = []
        
        for i in range(num_queries):
            # 获取单个查询向量的所有结果
            d_row = merged_dists[i]
            i_row = merged_idxs[i]
            
            # 根据度量方式排序
            if self.metric == 'L2':
                # L2: 距离越小越好 -> 升序
                sorted_indices = np.argsort(d_row)
            else:
                # IP: 内积越大越好 -> 降序 (argsort 默认升序，所以对负值排序或反转)
                sorted_indices = np.argsort(d_row)[::-1]
                
            # 取前 k 个
            top_k_indices = sorted_indices[:k]
            
            final_dists.append(d_row[top_k_indices])
            final_idxs.append(i_row[top_k_indices])
            
        return np.array(final_dists), np.array(final_idxs)

    def remove_ids(self, ids):
        """
        路由删除：计算 ID 所在分片并删除
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
        """保存所有分片"""
        print("正在保存所有分片...")
        for i, shard in enumerate(self.shards):
            # index_file 已经在 init 中通过 ThreadSafeFaissManager 指定了
            # 这里我们需要确保 ThreadSafeFaissManager 记住了 path，或者我们重新传入
            # 你的原始类 save 需要传入文件名，所以这里我们重新构建路径
            path = os.path.join(self.db_path, f"shard_{i}.index")
            shard.save(path)
        print("所有分片保存完成。")

    def count(self):
        """统计总向量数"""
        total = 0
        for shard in self.shards:
            total += shard.count()
        return total