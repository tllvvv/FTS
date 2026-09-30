-- Подключение расширений, используемых в эксперименте.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS rum;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

SELECT extname, extversion FROM pg_extension ORDER BY 1;
