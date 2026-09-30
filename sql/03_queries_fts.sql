-- Полнотекстовые запросы. :q -- параметр (текст tsquery).

--одиночная лексема (высоко-, средне- и низкочастотная)
SELECT count(*) FROM docs_large
WHERE to_tsvector('english', text) @@ to_tsquery('english', :'q');

--конъюнкция двух лексем
SELECT count(*) FROM docs_large
WHERE to_tsvector('english', text) @@ to_tsquery('english', :'q');

--фразовый запрос
SELECT count(*) FROM docs_large
WHERE to_tsvector('english', text) @@ to_tsquery('english', :'q');

--десять наиболее релевантных документов
SELECT id FROM docs_large
WHERE to_tsvector('english', text) @@ to_tsquery('english', :'q')
ORDER BY ts_rank(to_tsvector('english', text), to_tsquery('english', :'q')) DESC
LIMIT 10;

--то же через оператор расстояния RUM (вычисляется по индексу)
SELECT id FROM docs_large
WHERE to_tsvector('english', text) @@ to_tsquery('english', :'q')
ORDER BY to_tsvector('english', text) <=> to_tsquery('english', :'q')
LIMIT 10;
