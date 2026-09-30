import argparse
import csv
import json
import time

import numpy as np
import psycopg2

from common import DSN, RESULTS, DATA_PREP, SIZES

RAW = RESULTS / "raw_measurements.csv"
PLANS = RESULTS / "plans.jsonl"

FIELDS = ["experiment", "table", "rows", "index", "query", "param",
          "run", "seconds", "result_rows", "extra"]

# ---------------------------------------------------------------- индексы --
INDEXES = {
    "fts_gin":   ("fts",    "CREATE INDEX {n} ON {t} USING gin (to_tsvector('english', text))"),
    "fts_gist":  ("fts",    "CREATE INDEX {n} ON {t} USING gist (to_tsvector('english', text))"),
    "fts_rum":   ("fts",    "CREATE INDEX {n} ON {t} USING rum (to_tsvector('english', text) rum_tsvector_ops)"),
    "trgm_gin":  ("trgm",   "CREATE INDEX {n} ON {t} USING gin (text gin_trgm_ops)"),
    "trgm_gist": ("trgm",   "CREATE INDEX {n} ON {t} USING gist (text gist_trgm_ops)"),
    "vec_hnsw":  ("vector", "CREATE INDEX {n} ON {t} USING hnsw (emb vector_cosine_ops) WITH (m = 16, ef_construction = 64)"),
    "vec_ivfflat": ("vector", "CREATE INDEX {n} ON {t} USING ivfflat (emb vector_cosine_ops) WITH (lists = {lists})"),
    # вариант GIN с отключённым отложенным списком -- только для эксперимента insert
    "fts_gin_nofu": ("fts", "CREATE INDEX {n} ON {t} USING gin (to_tsvector('english', text)) WITH (fastupdate = off)"),
}

ANN_PARAMS = {
    "vec_hnsw":    [("hnsw.ef_search", v) for v in (10, 40, 100, 200)],
    "vec_ivfflat": [("ivfflat.probes", v) for v in (1, 5, 10, 20)],
}


def q_fts(table):
    base = ("SELECT count(*) FROM {t} WHERE to_tsvector('english', text) "
            "@@ to_tsquery('english', %s)").format(t=table)
    rank = ("SELECT id FROM {t} WHERE to_tsvector('english', text) @@ to_tsquery('english', %s) "
            "ORDER BY ts_rank(to_tsvector('english', text), to_tsquery('english', %s)) DESC "
            "LIMIT 10").format(t=table)
    rum = ("SELECT id FROM {t} WHERE to_tsvector('english', text) @@ to_tsquery('english', %s) "
           "ORDER BY to_tsvector('english', text) <=> to_tsquery('english', %s) "
           "LIMIT 10").format(t=table)
    return base, rank, rum


def q_trgm(table):
    like = "SELECT count(*) FROM {t} WHERE text LIKE %s".format(t=table)
    wsim = "SELECT count(*) FROM {t} WHERE %s <%% text".format(t=table)
    wtop = ("SELECT id FROM {t} WHERE %s <%% text "
            "ORDER BY %s <<-> text LIMIT 10").format(t=table)
    return like, wsim, wtop


def q_vec(table):
    return "SELECT id FROM {t} ORDER BY emb <=> %s::vector LIMIT 10".format(t=table)


# ------------------------------------------------------------- утилиты БД --
def connect():
    con = psycopg2.connect(**DSN)
    con.autocommit = True
    return con


def timed(cur, sql, params=None):
    t0 = time.perf_counter()
    cur.execute(sql, params)
    rows = []
    n = 0
    if cur.description is not None:
        rows = cur.fetchall()
        n = len(rows)
        if n == 1 and len(rows[0]) == 1 and isinstance(rows[0][0], int):
            n = rows[0][0]
    return time.perf_counter() - t0, n, rows


def explain(cur, sql, params=None):
    cur.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql, params)
    return cur.fetchone()[0]


def plan_nodes(plan):
    """Типы узлов плана: нужны, чтобы установить, был ли задействован индекс."""
    out = []
    stack = [plan[0]["Plan"]]
    while stack:
        nd = stack.pop()
        label = nd.get("Node Type", "")
        if "Index Name" in nd:
            label += ":" + nd["Index Name"]
        out.append(label)
        stack.extend(nd.get("Plans", []))
    return out


class Writer(object):
    def __init__(self):
        new = not RAW.exists()
        self.f = open(RAW, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=FIELDS)
        if new:
            self.w.writeheader()

    def add(self, **kw):
        self.w.writerow(dict((k, kw.get(k, "")) for k in FIELDS))
        self.f.flush()

    def done(self):
        self.f.close()


def log_plan(tag, plan):
    with open(PLANS, "a", encoding="utf-8") as f:
        f.write(json.dumps({"tag": tag, "plan": plan}, ensure_ascii=False) + "\n")


def index_size(cur, name):
    cur.execute("SELECT pg_relation_size(%s::regclass)", (name,))
    return cur.fetchone()[0]


def drop_index(cur, name):
    cur.execute("DROP INDEX IF EXISTS {n}".format(n=name))


def lists_for(rows):
    return max(10, rows // 1000)


# -------------------------------------------------------------- измерения --
def measure_build(cur, w, table, rows, idx, repeats):
    """Время построения индекса; после последнего повтора индекс остаётся в базе."""
    name = "{t}_{i}".format(t=table, i=idx)
    ddl = INDEXES[idx][1].format(n=name, t=table, lists=lists_for(rows))
    size = 0
    for r in range(1, repeats + 1):
        drop_index(cur, name)
        cur.execute("CHECKPOINT")
        t0 = time.perf_counter()
        cur.execute(ddl)
        el = time.perf_counter() - t0
        size = index_size(cur, name)
        w.add(experiment="build", table=table, rows=rows, index=idx,
              run=r, seconds=round(el, 4), extra=size)
        print("    build  {:<13} run {}: {:8.2f} c, {:>11} B".format(idx, r, el, size), flush=True)
    w.add(experiment="size", table=table, rows=rows, index=idx, run=1,
          seconds=0, result_rows=size, extra=size)
    return name


def measure_search(cur, w, table, rows, idx, queries, repeats, qemb, only=None):
    """Время поиска для семейства запросов, соответствующего индексу."""
    family = INDEXES[idx][0] if idx in INDEXES else "all"
    fams = ("fts", "trgm", "vector") if family == "all" else (family,)
    if only:
        fams = tuple(f for f in fams if f in only)

    for fam in fams:
        if fam == "fts":
            base, rank, rum = q_fts(table)
            items = [("fts_" + k, base, (queries["fts"][k],)) for k in
                     ("high", "mid", "low", "conj", "phrase")]
            items.append(("fts_rank_top10", rank, (queries["fts"]["mid"],) * 2))
            if idx in ("fts_rum", "none"):
                items.append(("fts_rum_dist_top10", rum, (queries["fts"]["mid"],) * 2))
        elif fam == "trgm":
            like, wsim, wtop = q_trgm(table)
            items = [
                ("trgm_like_common", like,
                 ("%" + queries["trgm"]["substring"]["trgm_common"] + "%",)),
                ("trgm_like_rare", like,
                 ("%" + queries["trgm"]["substring"]["trgm_rare"] + "%",)),
                ("trgm_wordsim", wsim, (queries["trgm"]["fuzzy"]["trgm_word"],)),
                ("trgm_typo_top10", wtop, (queries["trgm"]["fuzzy"]["trgm_typo"],) * 2),
            ]
        else:
            items = [("vec_knn10", q_vec(table), None)]

        for qid, sql, params in items:
            param_sets = [("", "")]
            if fam == "vector" and idx in ANN_PARAMS:
                param_sets = ANN_PARAMS[idx]
            for pname, pval in param_sets:
                if pname:
                    cur.execute("SET {p} = {v}".format(p=pname, v=pval))
                if fam == "vector":
                    run_vector(cur, w, table, rows, idx, sql, qemb, repeats,
                               "{p}={v}".format(p=pname, v=pval) if pname else "")
                    continue
                timed(cur, sql, params)                        # прогрев
                try:
                    pl = explain(cur, sql, params)
                    log_plan("{t}|{i}|{q}".format(t=table, i=idx, q=qid), pl)
                    nodes = ";".join(plan_nodes(pl))
                except Exception as exc:
                    nodes = "explain-failed: {}".format(exc)
                el = 0.0
                for r in range(1, repeats + 1):
                    el, n, _ = timed(cur, sql, params)
                    w.add(experiment="search", table=table, rows=rows, index=idx,
                          query=qid, param="", run=r, seconds=round(el, 6),
                          result_rows=n, extra=nodes if r == 1 else "")
                print("      {:<20} {:<13} {:>9.4f} c".format(qid, idx, el), flush=True)


def run_vector(cur, w, table, rows, idx, sql, qemb, repeats, param):
    """kNN-поиск: время усредняется по 20 запросам, дополнительно считается recall@10."""
    exact = np.load(str(DATA_PREP / "groundtruth_{t}.npy".format(t=table)))
    lits = ["[" + ",".join("%.6f" % v for v in q) + "]" for q in qemb]
    timed(cur, sql, (lits[0],))                                # прогрев
    tot = 0.0
    rec = 0.0
    for r in range(1, repeats + 1):
        tot = 0.0
        hits = 0
        for i, lit in enumerate(lits):
            el, _, res = timed(cur, sql, (lit,))
            tot += el
            hits += len(set(x[0] for x in res) & set(exact[i].tolist()))
        rec = hits / (10.0 * len(lits))
        w.add(experiment="search", table=table, rows=rows, index=idx,
              query="vec_knn10", param=param, run=r,
              seconds=round(tot / len(lits), 6), result_rows=10,
              extra="recall@10={:.4f}".format(rec))
        if r == 1:
            pl = explain(cur, sql, (lits[0],))
            log_plan("{t}|{i}|vec_knn10|{p}".format(t=table, i=idx, p=param), pl)
    print("      {:<20} {:<13} {:>9.4f} c  recall@10={:.3f}".format(
        "vec_knn10 " + param, idx, tot / len(lits), rec), flush=True)


def measure_insert(cur, w, table, rows, idx, repeats):
    """Массовая вставка 1 000 строк в таблицу с уже построенным индексом."""
    sql = ("INSERT INTO {t} (id, text, l1, l2, l3, emb) "
           "SELECT id, text, l1, l2, l3, emb FROM docs_insert").format(t=table)
    for r in range(1, repeats + 1):
        cur.execute("DELETE FROM {t} WHERE id IN (SELECT id FROM docs_insert)".format(t=table))
        cur.execute("VACUUM {t}".format(t=table))
        cur.execute("CHECKPOINT")
        t0 = time.perf_counter()
        cur.execute(sql)
        el = time.perf_counter() - t0
        w.add(experiment="insert", table=table, rows=rows, index=idx,
              run=r, seconds=round(el, 4), result_rows=1000)
        print("    insert {:<13} run {}: {:8.3f} c".format(idx, r, el), flush=True)
    cur.execute("DELETE FROM {t} WHERE id IN (SELECT id FROM docs_insert)".format(t=table))
    cur.execute("VACUUM {t}".format(t=table))


def build_groundtruth(cur, table, qemb):
    """Точные ответы kNN (полный перебор) -- эталон для оценки полноты."""
    path = DATA_PREP / "groundtruth_{t}.npy".format(t=table)
    if path.exists():
        return
    res = []
    for q in qemb:
        lit = "[" + ",".join("%.6f" % v for v in q) + "]"
        cur.execute("SELECT id FROM {t} ORDER BY emb <=> %s::vector LIMIT 10".format(t=table),
                    (lit,))
        res.append([x[0] for x in cur.fetchall()])
    np.save(str(path), np.array(res, dtype=np.int64))
    print("    эталон kNN сохранён:", path.name, flush=True)


# ------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="small,medium,large")
    ap.add_argument("--indexes", default=",".join(list(INDEXES.keys()) + ["none"]))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--build-repeats-large", type=int, default=3)
    ap.add_argument("--skip", default="", help="через запятую: build,search,insert")
    ap.add_argument("--families", default="", help="через запятую: fts,trgm,vector")
    args = ap.parse_args()

    skip = set(x for x in args.skip.split(",") if x)
    fams_only = set(x for x in args.families.split(",") if x)
    wanted = [x for x in args.indexes.split(",") if x]
    with open(str(RESULTS / "queries.json"), encoding="utf-8") as f:
        queries = json.load(f)
    qemb = np.load(str(DATA_PREP / "query_embeddings.npy"))

    con = connect()
    cur = con.cursor()
    cur.execute("SET jit = off")
    w = Writer()

    for key in args.sizes.split(","):
        rows = SIZES[key]
        table = "docs_{k}".format(k=key)
        print("\n=== {t} ({r} строк) ===".format(t=table, r=rows), flush=True)
        cur.execute("ANALYZE {t}".format(t=table))
        build_groundtruth(cur, table, qemb)

        if "none" in wanted:
            print("  -- без индекса", flush=True)
            if "search" not in skip:
                measure_search(cur, w, table, rows, "none", queries, args.repeats, qemb, fams_only)
            if "insert" not in skip:
                measure_insert(cur, w, table, rows, "none", args.repeats)

        for idx in wanted:
            if idx not in INDEXES:
                continue
            print("  -- {i}".format(i=idx), flush=True)
            br = args.build_repeats_large if key == "large" else args.repeats
            name = "{t}_{i}".format(t=table, i=idx)
            if "build" in skip:
                drop_index(cur, name)
                cur.execute(INDEXES[idx][1].format(n=name, t=table, lists=lists_for(rows)))
            else:
                measure_build(cur, w, table, rows, idx, br)
            cur.execute("ANALYZE {t}".format(t=table))
            if "search" not in skip and idx != "fts_gin_nofu":
                measure_search(cur, w, table, rows, idx, queries, args.repeats, qemb, fams_only)
            if "insert" not in skip:
                measure_insert(cur, w, table, rows, idx, min(args.repeats, br))
            drop_index(cur, name)
            cur.execute("VACUUM {t}".format(t=table))

    w.done()
    con.close()
    print("\nГотово. Результаты:", RAW)


if __name__ == "__main__":
    main()
