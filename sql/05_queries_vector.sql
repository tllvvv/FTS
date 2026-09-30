-- Семантический поиск: k ближайших соседей по косинусному расстоянию.
-- :q -- литерал вектора вида

-- Параметры, управляющие компромиссом «время--полнота»
SET hnsw.ef_search = 40;   
SET ivfflat.probes = 1; 

SELECT id FROM docs_large ORDER BY emb <=> :'q'::vector LIMIT 10;

-- Эталон (полный перебор)
BEGIN;
SET LOCAL enable_indexscan = off;
SELECT id FROM docs_large ORDER BY emb <=> :'q'::vector LIMIT 10;
COMMIT;
