-- Частотный словарь лексем
SELECT word, ndoc, nentry
FROM ts_stat($$SELECT to_tsvector('english', text) FROM docs_medium$$)
WHERE length(word) >= 5 AND word ~ '^[a-z]+$'
ORDER BY ndoc DESC
LIMIT 30;
