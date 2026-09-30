import csv
import io
import sys
import time

import numpy as np
import pandas as pd
import psycopg2

from common import DATA_PREP, DSN, SIZES, INSERT_BATCH, EMB_DIM

CHUNK = 5_000

DDL = """
CREATE TABLE IF NOT EXISTS {t} (
    id    integer PRIMARY KEY,
    text  text NOT NULL,
    l1    text,
    l2    text,
    l3    text,
    emb   vector({d})
);
"""


def load_corpus():
    p = DATA_PREP / "corpus.parquet"
    return pd.read_parquet(p) if p.exists() else pd.read_csv(DATA_PREP / "corpus.csv")


def vec_literal(row):
    return "[" + ",".join("%.6f" % v for v in row) + "]"


def copy_rows(cur, table, df, emb):
    total = len(df)
    t0 = time.perf_counter()
    for start in range(0, total, CHUNK):
        sub = df.iloc[start:start + CHUNK]
        sub_emb = emb[start:start + CHUNK]
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        for (_, r), e in zip(sub.iterrows(), sub_emb):
            w.writerow([r.id, r.text, r.l1, r.l2, r.l3, vec_literal(e)])
        buf.seek(0)
        cur.copy_expert(
            "COPY {} (id, text, l1, l2, l3, emb) FROM STDIN WITH (FORMAT csv)".format(table),
            buf)
        if (start // CHUNK) % 10 == 0 and start:
            el = time.perf_counter() - t0
            print("    {}: {}/{} ({:.0f} стр/с)".format(table, start, total, start / el), flush=True)
    print("    {}: загружено {} строк за {:.1f} с".format(table, total, time.perf_counter() - t0))


def main():
    admin = dict(DSN)
    dbname = admin.pop("dbname")
    con = psycopg2.connect(dbname="postgres", **admin)
    con.autocommit = True
    with con.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (dbname,))
        if cur.fetchone():
            if "--force" not in sys.argv:
                sys.exit("База {} уже существует. Запустите с --force для пересоздания.".format(dbname))
            cur.execute('DROP DATABASE "{}" WITH (FORCE)'.format(dbname))
        cur.execute('CREATE DATABASE "{}" ENCODING \'UTF8\''.format(dbname))
    con.close()

    con = psycopg2.connect(**DSN)
    con.autocommit = True
    cur = con.cursor()
    for ext in ("vector", "rum", "pg_trgm"):
        cur.execute("CREATE EXTENSION IF NOT EXISTS {}".format(ext))
    cur.execute("SELECT extname, extversion FROM pg_extension ORDER BY 1")
    print("Расширения:", cur.fetchall())

    df = load_corpus()
    emb = np.load(DATA_PREP / "embeddings.npy", mmap_mode="r")
    if len(emb) != len(df):
        sys.exit("Размеры corpus и embeddings не совпадают: {} и {}".format(len(df), len(emb)))

    corpus = df[df.part == "corpus"].reset_index(drop=True)
    ins = df[df.part == "insert"].reset_index(drop=True)
    emb_corpus = emb[:len(corpus)]
    emb_ins = emb[len(corpus):]

    plan = [("docs_{}".format(k), corpus.iloc[:n], emb_corpus[:n]) for k, n in SIZES.items()]
    plan.append(("docs_insert", ins, emb_ins))

    for table, sub, e in plan:
        cur.execute("DROP TABLE IF EXISTS {}".format(table))
        cur.execute(DDL.format(t=table, d=EMB_DIM))
        copy_rows(cur, table, sub, e)
        cur.execute("ANALYZE {}".format(table))

    cur.execute("""SELECT relname, n_live_tup, pg_size_pretty(pg_total_relation_size(relid))
                   FROM pg_stat_user_tables ORDER BY relname""")
    for row in cur.fetchall():
        print("  {:<14} {:>8} строк  {}".format(*row))
    con.close()
    print("Загрузка завершена, ожидаемо {} + {} строк".format(SIZES["large"], INSERT_BATCH))


if __name__ == "__main__":
    main()
