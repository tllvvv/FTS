import json

import numpy as np
import psycopg2

from common import DSN, RESULTS, DATA_PREP, ROOT, EMB_DIM, MAX_TOKENS

# Естественно-языковые запросы для векторного поиска
NL_QUERIES = [
    "a medieval castle in southern Germany",
    "professional basketball player who played in the NBA",
    "a species of moth found in South America",
    "railway station serving a small town in Japan",
    "romantic comedy film released in the 1990s",
    "heavy metal band formed in Scandinavia",
    "river flowing through central Africa",
    "private secondary school for girls",
    "Roman Catholic church building in Italy",
    "military officer who fought in the Second World War",
    "studio album by an American hip hop artist",
    "village in the north of India",
    "flowering plant of the orchid family",
    "association football club playing in a regional league",
    "scientific journal on organic chemistry",
    "historic lighthouse on the Atlantic coast",
    "politician elected to a national parliament",
    "computer video game for home consoles",
    "mountain peak in the Andes",
    "novel written by a British author",
]

# Подстрочные и нечёткие запросы для pg_trgm
TRGM_SUBSTRINGS = {"trgm_common": "university", "trgm_rare": "chromatogra"}
TRGM_FUZZY = {"trgm_typo": "univercity", "trgm_word": "footballer"}


def pick_lexemes(cur):
    """Частотный словарь по выборке 30 000 статей."""
    cur.execute("""
        SELECT word, ndoc
        FROM ts_stat($$SELECT to_tsvector('english', text) FROM docs_medium$$)
        WHERE length(word) >= 5 AND word ~ '^[a-z]+$'
        ORDER BY ndoc DESC
    """)
    rows = cur.fetchall()
    n = 30000
    used = set()

    def nearest(target):
        cand = min((r for r in rows if r[0] not in used),
                   key=lambda r: abs(r[1] / n - target))
        used.add(cand[0])
        return cand

    high = nearest(0.10)
    mid = nearest(0.01)
    low = nearest(0.0005)
    return {"fts_high": high[0], "fts_mid": mid[0], "fts_low": low[0]}, rows


def doc_freq(cur, table, lexeme):
    cur.execute("SELECT count(*) FROM {} WHERE to_tsvector('english', text) @@ "
                "to_tsquery('english', %s)".format(table), (lexeme,))
    return cur.fetchone()[0]


def main():
    con = psycopg2.connect(**DSN)
    con.autocommit = True
    cur = con.cursor()

    lex, rows = pick_lexemes(cur)
    print("Выбранные лексемы:", lex)

    # Фраза: пара подряд идущих слов, реально встречающаяся в корпусе
    phrase = "united <-> states"
    conj = "{} & {}".format(lex["fts_high"], lex["fts_mid"])

    queries = {
        "fts": {
            "high": lex["fts_high"],
            "mid": lex["fts_mid"],
            "low": lex["fts_low"],
            "conj": conj,
            "phrase": phrase,
        },
        "trgm": {"substring": TRGM_SUBSTRINGS, "fuzzy": TRGM_FUZZY},
        "vector": {"nl_queries": NL_QUERIES},
        "doc_freq": {},
    }

    for table in ("docs_small", "docs_medium", "docs_large"):
        queries["doc_freq"][table] = {
            k: doc_freq(cur, table, v) for k, v in
            (("high", lex["fts_high"]), ("mid", lex["fts_mid"]), ("low", lex["fts_low"]))
        }
        for k, q in (("conj", conj), ("phrase", phrase)):
            cur.execute("SELECT count(*) FROM {} WHERE to_tsvector('english', text) @@ "
                        "to_tsquery('english', %s)".format(table), (q,))
            queries["doc_freq"][table][k] = cur.fetchone()[0]
        print(table, queries["doc_freq"][table])

    # Топ-30 лексем -- для приложения к отчёту
    with open(RESULTS / "lexeme_freq_top30.csv", "w", encoding="utf-8") as f:
        f.write("word,ndoc\n")
        for w, nd in rows[:30]:
            f.write("{},{}\n".format(w, nd))

    # Эмбеддинги естественно-языковых запросов
    import onnxruntime as ort
    from tokenizers import Tokenizer
    M = ROOT / "models" / "all-MiniLM-L6-v2"
    tok = Tokenizer.from_file(str(M / "tokenizer.json"))
    tok.enable_truncation(max_length=MAX_TOKENS)
    tok.enable_padding(pad_id=0, pad_token="[PAD]")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(str(M / "onnx" / "model.onnx"), so,
                                providers=["CPUExecutionProvider"])
    enc = tok.encode_batch(NL_QUERIES)
    ids = np.array([e.ids for e in enc], dtype=np.int64)
    mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
    h = sess.run(None, {"input_ids": ids, "attention_mask": mask,
                        "token_type_ids": np.zeros_like(ids)})[0]
    m = mask[:, :, None].astype(np.float32)
    p = (h * m).sum(1) / m.sum(1)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    np.save(DATA_PREP / "query_embeddings.npy", p.astype(np.float32))
    print("Сохранено эмбеддингов запросов:", p.shape)

    with open(RESULTS / "queries.json", "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    print("Сохранено:", RESULTS / "queries.json")


if __name__ == "__main__":
    main()
