-- Схема таблиц.
CREATE TABLE docs_small (
    id    integer PRIMARY KEY,
    text  text NOT NULL,
    l1    text,            
    l2    text,
    l3    text,
    emb   vector(384)      -- эмбеддинг all-MiniLM-L6-v2, L2-нормализованный
);

CREATE TABLE docs_medium (LIKE docs_small INCLUDING ALL);
CREATE TABLE docs_large  (LIKE docs_small INCLUDING ALL);
CREATE TABLE docs_insert (LIKE docs_small INCLUDING ALL);
