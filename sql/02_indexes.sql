-- Определения всех рассматриваемых индексов (на примере docs_large).

-- Полнотекстовые индексы по выражению to_tsvector
CREATE INDEX docs_large_fts_gin  ON docs_large USING gin  (to_tsvector('english', text));
CREATE INDEX docs_large_fts_gist ON docs_large USING gist (to_tsvector('english', text));
CREATE INDEX docs_large_fts_rum  ON docs_large USING rum  (to_tsvector('english', text) rum_tsvector_ops);

-- Вариант GIN с отключённым отложенным списком (эксперимент со вставкой)
CREATE INDEX docs_large_fts_gin_nofu ON docs_large
    USING gin (to_tsvector('english', text)) WITH (fastupdate = off);

-- Триграммные индексы
CREATE INDEX docs_large_trgm_gin  ON docs_large USING gin  (text gin_trgm_ops);
CREATE INDEX docs_large_trgm_gist ON docs_large USING gist (text gist_trgm_ops);

-- Векторные индексы (косинусное расстояние)
CREATE INDEX docs_large_vec_hnsw ON docs_large
    USING hnsw (emb vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX docs_large_vec_ivfflat ON docs_large
    USING ivfflat (emb vector_cosine_ops) WITH (lists = 300);

-- Размеры построенных индексов
SELECT indexrelname, pg_size_pretty(pg_relation_size(indexrelid))
FROM pg_stat_user_indexes WHERE relname = 'docs_large' ORDER BY 1;
