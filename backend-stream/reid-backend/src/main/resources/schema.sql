-- 1. 구역+CCTV 필터링용 B-Tree 인덱스
CREATE INDEX IF NOT EXISTS idx_reid_location_cctv
ON reid_result (location_id, cctv_id);

-- 2. 벡터 검색용 HNSW 인덱스
CREATE INDEX IF NOT EXISTS idx_reid_vector_hnsw
ON reid_result USING hnsw (vector_data vector_cosine_ops)
WITH (m = 16, ef_construction = 64);