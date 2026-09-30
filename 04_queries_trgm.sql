-- Подстрочный и нечёткий поиск (pg_trgm).

--поиск подстроки
SELECT count(*) FROM docs_large WHERE text LIKE '%university%';
SELECT count(*) FROM docs_large WHERE text LIKE '%chromatogra%';

--словное сходство
SELECT count(*) FROM docs_large WHERE 'footballer' <% text;

-- десять документов, наиболее похожих на слово с опечаткой;
SELECT id FROM docs_large
WHERE 'univercity' <% text
ORDER BY 'univercity' <<-> text
LIMIT 10;

SHOW pg_trgm.similarity_threshold;
SHOW pg_trgm.word_similarity_threshold;
